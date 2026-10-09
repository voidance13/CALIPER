"""Read frozen predictions; audit Table 3 and select a current misrouting case.

No generation, cloud calls, model training, threshold tuning or old-file writes.
"""
import argparse
import hashlib
import json
import re
import string
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import fasttext
import jieba

BASE = Path('/mnt/data/caliper/code/CALIPER')
NEW = BASE / 'classifier_analysis/overlap_free_retrain_20260927'
THRESHOLDS = {'hotpotqa': .974, 'musique': .978, '2wikimultihopqa': .977}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(text):
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in text.lower()).split())


def norm(text):
    text = re.sub(r'\b(a|an|the)\b', ' ', str(text).lower())
    return ' '.join(''.join(c for c in text if c not in string.punctuation).split())


def answer(text):
    for marker in ('Answer:\n', 'answer:\n', 'Answer: ', 'answer: ', 'Answer:', 'answer:'):
        match = re.search(re.escape(marker), text or '')
        if match:
            return text[match.end():]
    line = (text or '').split('\n')[-1]
    match = re.search(': ', line)
    return line[match.end():] if match else line


def f1(prediction, gold):
    p, g = norm(answer(prediction)).split(), norm(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    precision = common / len(p) if p else 0
    recall = common / len(g) if g else 0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def first(rows):
    result = {}
    for row in rows:
        result.setdefault(row['question'], row)
    return result


def pred(row):
    return row.get('pred') or row.get('res', '')


def gold(row):
    return row['answer'] if isinstance(row.get('answer'), str) else (row.get('answers') or [''])[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    old_path = BASE / 'table-result/table3_full_unified_metrics.json'
    old = load(old_path)
    clf_path = NEW / '9b_clf_overlap_free.bin'
    clf = fasttext.load_model(str(clf_path))
    assert sha(clf_path) == '44d7ea0de0d214bce1e44f92b9452419838daec1e71ba38a9bef42e32806fd6d'
    stops_path = BASE / 'query_clf/stopwords.txt'
    stops = set(stops_path.read_text().splitlines())
    def probability(q):
        text = ' '.join(t for t in jieba.lcut(q) if len(t) > 1 and t not in stops)
        labels, probabilities = clf.predict(text, k=2)
        return float(dict(zip(labels, probabilities)).get('__label__easy', 0))

    split_questions = {}
    for split in ('train', 'valid'):
        path = NEW / f'9b_{split}_overlap_free.txt'
        split_questions[split] = {canonical(l.split(' ', 1)[1]) for l in path.read_text().splitlines()}
    report = {
        'status': 'AUDIT_NOT_FROZEN', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'no_generation_or_cloud_calls': True,
        'old_table3_sha256': sha(old_path), 'model_sha256': sha(clf_path),
        'thresholds': THRESHOLDS, 'datasets': {}, 'sources_sha256': {},
        'note': 'Full-pool classifier F1 is diagnostic only when overlap is nonzero; it is not an independent held-out result.'}
    for dataset, threshold in THRESHOLDS.items():
        test_path = BASE / f'data/{dataset}/test.json'
        tests = first(load(test_path))
        sources = {}
        for method, metadata in old['datasets'][dataset]['sources'].items():
            path = BASE / metadata['path']
            actual_sha = sha(path)
            assert actual_sha == metadata['sha256'], (dataset, method, 'source hash changed')
            sources[method] = first(load(path))
            assert set(sources[method]) == set(tests)
            value = sum(f1(pred(sources[method][q]), gold(tests[q])) for q in tests) / len(tests) * 100
            assert round(value, 2) == old['datasets'][dataset]['methods'][method]['F1'], (dataset, method, value)
            report['sources_sha256'][str(path)] = actual_sha
        overlap = {split: sorted(q for q in tests if canonical(q) in questions)
                   for split, questions in split_questions.items()}
        blocked = set(overlap['train'] + overlap['valid'])
        rows = []
        for q in tests:
            p = probability(q)
            method = 'Qwen3.5-9B' if p >= threshold else 'CALIPER'
            rows.append({'dataset': dataset, 'question': q, 'p_easy': p, 'threshold': threshold,
                         'route': method, 'gold': gold(tests[q]),
                         'local_pred': answer(pred(sources['Qwen3.5-9B'][q])),
                         'core_pred': answer(pred(sources['CALIPER'][q])),
                         'selected_F1': f1(pred(sources[method][q]), gold(tests[q])),
                         'overlap_with_fitting': q in blocked})
        with (args.output / f'table3_{dataset}_new_classifier_audit.jsonl').open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        heldout = [r for r in rows if not r['overlap_with_fitting']]
        report['datasets'][dataset] = {
            'n': len(rows), 'old_classifier_F1': old['datasets'][dataset]['methods']['CALIPER+clf']['F1'],
            'full_pool_new_classifier_F1_diagnostic': sum(r['selected_F1'] for r in rows) / len(rows) * 100,
            'fitting_overlap_questions': overlap, 'heldout_n': len(heldout),
            'heldout_new_classifier_F1': sum(r['selected_F1'] for r in heldout) / len(heldout) * 100,
            'local_n': sum(r['route'] == 'Qwen3.5-9B' for r in rows)}
        report['sources_sha256'][str(test_path)] = sha(test_path)
        print(dataset, json.dumps(report['datasets'][dataset]), flush=True)

    # Same declared selection rule: smallest SHA256('42|question') among
    # true-hard MuSiQue queries routed locally with core F1=1 and local F1=0.
    ds = 'musique'
    tests = first(load(BASE / f'data/{ds}/test.json'))
    baseline_path = NEW / 'frozen_source_cache/musique_baseline_frozen_table2.json'
    core_path = NEW / 'frozen_source_cache/musique_v1_frozen_table2.json'
    baseline, core = first(load(baseline_path)), first(load(core_path))
    route_path = NEW / 'table2_candidate_predictions_dataset_thresholds.jsonl'
    routed = [json.loads(line) for line in route_path.read_text().splitlines()]
    seed_paths = [BASE / f'query_clf/musique_base_sample{i}.json' for i in range(1, 4)]
    seeds = [dict(load(path)) for path in seed_paths]
    candidates = []
    for row in routed:
        q = row['question']
        if row['dataset'] != ds or row['route'] != 'local_BASE' or q not in baseline or q not in core:
            continue
        if not all(q in run for run in seeds):
            continue
        target = gold(tests[q])
        successes = [bool(norm(target)) and norm(target) in norm(run[q]) for run in seeds]
        if all(successes) or f1(pred(core[q]), target) != 1 or f1(pred(baseline[q]), target) != 0:
            continue
        candidates.append({'question': q, 'gold': target, 'p_easy': row['p_easy'],
                           'threshold': row['threshold'], 'route': row['route'],
                           'local_pred': answer(pred(baseline[q])), 'core_pred': answer(pred(core[q])),
                           'seed_local_gold_hits': successes,
                           'seed_local_outputs': [answer(run[q]) for run in seeds],
                           'selection_sha256': hashlib.sha256(('42|' + q).encode()).hexdigest()})
    assert candidates, 'No independently labeled misrouting case satisfies the declared rule'
    candidates.sort(key=lambda r: r['selection_sha256'])
    selected = candidates[0]
    selected['dataset'] = ds
    selected['candidate_count'] = len(candidates)
    selected['model_sha256'] = sha(clf_path)
    selected['selection_rule'] = "Minimum SHA256('42|question'); true-hard (not all three local runs recover gold), local route, core F1=1, local F1=0"
    selected['sources_sha256'] = {str(p): sha(p) for p in [baseline_path, core_path, route_path] + seed_paths}
    (args.output / 'selected_misrouting_case.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    (args.output / 'table3_classifier_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('SELECTED_CASE', json.dumps(selected, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
