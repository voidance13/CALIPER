"""Non-destructive, four-record supporting-fact repair and ranking-only rescore.

Does not train, retrieve, generate answers, change current pointers or overwrite
historical data. Table 3 generation remains explicitly pending.
"""
import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

BASE = Path('/mnt/data/caliper/code/CALIPER')
MIRROR = Path('/mnt/data/caliper/hf_cache/hub/datasets--framolfese--2WikiMultihopQA/snapshots/fe713bfbd1afbca1a65246741a75890405d56a3a/data/validation-00000-of-00001.parquet')
CORRECTIONS = {
    'b728e70c094411ebbdaeac1f6bf848b6': ('Dead man', 'Dead Man'),
    'd1aa7f57087e11ebbd6bac1f6bf848b6': ('Two wrongs make a right', 'Two Wrongs Make a Right'),
    '43b55fee087311ebbd66ac1f6bf848b6': ('Honeymoon suite', 'Honeymoon Suite'),
    '59acfe98096c11ebbdafac1f6bf848b6': ('Little black book', 'Little Black Book'),
}
EXPECTED_OLD = {'BASE': [62.61, 72.00, 85.42], 'FT-SEP': [89.35, 98.04, 99.46],
                'FT-MER': [84.49, 95.09, 98.83], 'CALIPER (routed)': [89.30, 98.00, 99.42]}
EXPECTED_NEW = {'BASE': [62.62, 72.01, 85.42], 'FT-SEP': [89.37, 98.05, 99.47],
                'FT-MER': [84.50, 95.10, 98.84], 'CALIPER (routed)': [89.32, 98.02, 99.43]}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def recall(row, ranked, k):
    context = {ctx['title']: ctx['sentences'] for ctx in row['ctxs']}
    gold = [context[title][index] for title, index in row['supporting_facts']]
    assert gold
    return sum(sentence in ranked[:k] for sentence in gold) / len(gold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = BASE / 'data/2wikimultihopqa/test.json'
    original_sha = sha(original)
    assert original_sha == '5d4aa10f016ee02994ebfaf321dcc309c2a23894e20902fe61fb9eb6bed38ad5'
    mirror_sha = sha(MIRROR)
    assert mirror_sha == '408e2dbb28edc6c8b9ca3ba0c94d4fc7bf17ffb923766593a3a7f546ab4cba59'
    old = load(original)
    mirror = pq.read_table(MIRROR).to_pylist()
    assert len(old) == len(mirror) == 12576
    mirror_by_id = {row['id']: row for row in mirror}
    assert len(mirror_by_id) == len(old)
    assert len({row['question'] for row in old}) == len(old)
    corrected = copy.deepcopy(old)
    patches = []
    for before, after in zip(old, corrected):
        ref = mirror_by_id[before['id']]
        assert before['question'] == ref['question']
        assert before['answers'] == [ref['answer']]
        assert before['type'] == ref['type']
        reference_ctx = [{'title': t, 'sentences': s} for t, s in zip(ref['context']['title'], ref['context']['sentences'])]
        assert before['ctxs'] == reference_ctx
        reference_sf = [list(pair) for pair in zip(ref['supporting_facts']['title'], ref['supporting_facts']['sent_id'])]
        if before['supporting_facts'] != reference_sf:
            assert before['id'] in CORRECTIONS
            wrong, right = CORRECTIONS[before['id']]
            expected = [[right if title == wrong else title, index] for title, index in before['supporting_facts']]
            assert expected == reference_sf
            assert sum(title == wrong for title, index in before['supporting_facts']) == 1
            after['supporting_facts'] = reference_sf
            index = next(index for title, index in before['supporting_facts'] if title == wrong)
            context = {c['title']: c['sentences'] for c in before['ctxs']}
            patches.append({'id': before['id'], 'question': before['question'],
                            'old_supporting_facts': before['supporting_facts'],
                            'new_supporting_facts': reference_sf,
                            'old_sentence': context[wrong][index], 'new_sentence': context[right][index]})
        assert {k: v for k, v in before.items() if k != 'supporting_facts'} == {k: v for k, v in after.items() if k != 'supporting_facts'}
    assert {p['id'] for p in patches} == set(CORRECTIONS)
    assert sum(a != b for a, b in zip(old, corrected)) == 4
    inputs = {str(original): original_sha, str(MIRROR): mirror_sha, str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    rank_paths = {
        'BASE': [BASE / 'data/2wikimultihopqa/test_retrieved.json'],
        'FT-SEP': [BASE / 'retrieval_runs/hp_tw_large_20260921/2wikimultihopqa_test_retrieved_ftsep_large.json'],
        'FT-MER': [BASE / 'data/2wikimultihopqa/test_retrieved_mergedft_rerankerft.json'],
        'CALIPER (routed)': [BASE / f'retrieval_runs/routed_large_20260921/shard_{i}.jsonl' for i in range(4)],
    }
    results = {}
    affected_scores = []
    for method, paths in rank_paths.items():
        ranked = {}
        for path in paths:
            inputs[str(path)] = sha(path)
            rows = load(path) if path.suffix == '.json' else [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for row in rows:
                if method == 'CALIPER (routed)' and row['dataset'] != 'tw':
                    continue
                assert row['question'] not in ranked
                assert not row.get('error')
                ranked[row['question']] = row['retrieved_text']
        assert set(ranked) == {row['question'] for row in old}
        before_values = [[recall(row, ranked[row['question']], k) for k in (3, 5, 10)] for row in old]
        after_values = [[recall(row, ranked[row['question']], k) for k in (3, 5, 10)] for row in corrected]
        before_mean = [math.fsum(v[j] for v in before_values) / len(old) * 100 for j in range(3)]
        after_mean = [math.fsum(v[j] for v in after_values) / len(old) * 100 for j in range(3)]
        assert [round(v, 2) for v in before_mean] == EXPECTED_OLD[method]
        assert [round(v, 2) for v in after_mean] == EXPECTED_NEW[method]
        results[method] = {'n': len(old), 'old_R3_R5_R10': before_mean, 'new_R3_R5_R10': after_mean,
                           'old_display': EXPECTED_OLD[method], 'new_display': EXPECTED_NEW[method],
                           'delta_percentage_points': [b - a for a, b in zip(before_mean, after_mean)]}
        for row, before, after in zip(old, before_values, after_values):
            if row['id'] in CORRECTIONS:
                affected_scores.append({'method': method, 'id': row['id'], 'question': row['question'],
                                        'old_R3_R5_R10': before, 'new_R3_R5_R10': after})
    assert sha(original) == original_sha
    args.output.mkdir(parents=True, exist_ok=False)
    corrected_path = args.output / 'test_supporting_facts_corrected.json'
    write(corrected_path, corrected)
    write(args.output / 'four_corrected_questions.json', [r for r in corrected if r['id'] in CORRECTIONS])
    write(args.output / 'supporting_fact_patch.json', patches)
    write(args.output / 'table7_recall_rescore.json', {'status': 'VERIFIED_RANKING_ONLY_RESCORE_PENDING_ADOPTION',
                                                    'metric': 'Exact sentence membership; per-question gold-sentence recall macro mean; k=3,5,10.',
                                                    'results': results, 'affected_scores': affected_scores})
    write(args.output / 'repair_manifest.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'ANNOTATION_COPY_AND_RECALL_REPAIR_COMPLETE_TABLE3_GENERATION_PENDING',
        'changed_record_count': 4, 'pool_n_unchanged': 12576,
        'reference_scope': 'Third-party cached framolfese validation mirror; not independent verification of original official release.',
        'input_sha256': inputs, 'output_sha256': {p.name: sha(p) for p in args.output.glob('*.json')},
        'original_unchanged': sha(original) == original_sha,
        'protocol_unchanged': ['question', 'answer', 'context', 'question order', 'evaluation pool', 'ranked retrieval results', 'recall definition'],
        'table3_status': 'Pending: recover actual historical prompts, model identity, decoding settings and reflection setting before regenerating only these four questions.',
        'current_manuscript_archive_pointers_changed': False,
    })
    print(json.dumps({'output': str(args.output), 'status': 'PASS_ANNOTATION_COPY_AND_RECALL_RESCORE', 'table3': 'PENDING', 'results': results}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
