"""Overlay exactly four authorized new answers; preserve historical sources.

Recompute same-pool Table 3 token F1 and existing paired-bootstrap contrasts.
No regeneration, training, threshold tuning, question selection or source writes.
"""
import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_table3_case_20260930 import BASE, f1, first, gold, load, pred, sha
from finalize_table3_case_20260930 import paired

CURRENT = BASE / 'table-result/table3_current_classifier_20260930'
ANNOTATION = BASE / 'annotation_repairs/2wiki_sf_20261002'
GENERATION = BASE / 'annotation_repairs/2wiki_sf_generation_20261004_v3'
METHODS = ('Qwen3.5-9B', 'CALIPER', 'Qwen3.7-plus', 'CALIPER+clf')


def write(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = load(GENERATION / 'GENERATION_RECEIPT.json')
    assert receipt['status'] == 'PASS_NEW_FOUR_QUESTION_GENERATION'
    assert receipt['repaired_question_n'] == 4 and not receipt['historical_reproduction']
    for name, expected in receipt['output_sha256'].items():
        assert sha(GENERATION / name) == expected
    recipe = load(GENERATION / 'run_recipe.json')
    for path, expected in recipe['input_sha256'].items():
        assert sha(Path(path)) == expected
    annotation = load(ANNOTATION / 'repair_manifest.json')
    for path, expected in annotation['input_sha256'].items():
        assert sha(Path(path)) == expected
    replacements = load(GENERATION / 'replacement_predictions.json')
    corrected = first(load(ANNOTATION / 'test_supporting_facts_corrected.json'))
    subset = first(load(ANNOTATION / 'four_corrected_questions.json'))
    assert set(replacements) == set(subset) and len(replacements) == 4
    old = load(CURRENT / 'table3_current_f1.json')
    old_stats = load(CURRENT / 'table3_paired_bootstrap.json')
    original_meta = load(BASE / 'table-result/table3_full_unified_metrics.json')
    original_sources = {}
    for method, meta in original_meta['datasets']['2wikimultihopqa']['sources'].items():
        p = BASE / meta['path']
        assert sha(p) == meta['sha256']
        original_sources[method] = first(load(p))
    current_rows_path = CURRENT / 'table3_2wikimultihopqa_paired_scores.jsonl'
    rows = [json.loads(line) for line in current_rows_path.read_text().splitlines() if line.strip()]
    assert len(rows) == len({r['question'] for r in rows}) == len(corrected) == 12576
    assert {r['question'] for r in rows} == set(corrected)
    changed = []
    for row in rows:
        q = row['question']
        assert row['gold'] == gold(corrected[q])
        for method, source in original_sources.items():
            assert abs(f1(pred(source[q]), row['gold']) - row['F1'][method]) < 1e-12
        expected_branch = 'Qwen3.5-9B' if row['p_easy'] >= row['threshold'] else 'CALIPER'
        assert row['threshold'] == .977 and row['route'] == expected_branch
        assert row['F1']['CALIPER+clf'] == row['F1'][expected_branch]
        if q not in replacements:
            continue
        before = copy.deepcopy(row)
        assert row['route'] == 'CALIPER'
        assert set(replacements[q]) == set(original_sources)
        for method, replacement in replacements[q].items():
            assert replacement['question'] == q and replacement['answer'] == row['gold']
            assert replacement['id'] == subset[q]['id']
            row['F1'][method] = f1(pred(replacement), row['gold'])
        row['F1']['CALIPER+clf'] = row['F1']['CALIPER']
        changed.append({'id': subset[q]['id'], 'question': q, 'old_scores': before['F1'], 'new_scores': row['F1'],
                        'route': row['route'], 'threshold': row['threshold'], 'p_easy': row['p_easy'],
                        'old_predictions': {m: pred(s[q]) for m, s in original_sources.items()},
                        'new_predictions': {m: pred(r) for m, r in replacements[q].items()}})
    assert len(changed) == 4
    scores = {m: [r['F1'][m] for r in rows] for m in METHODS}
    new = copy.deepcopy(old)
    new.update({'status': 'AUTHORIZED_ANNOTATION_REPAIR_PENDING_PAPER_ARCHIVE_SYNC',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'protocol': 'Original retained Table 3 question set and scoring, with exactly four supporting-fact correction questions replaced by a documented new current-configuration run; no claim of historical reproduction.',
                'no_generation_cloud_training_or_threshold_tuning': False,
                'repair_scope': {'generated_questions': 4, 'training': False, 'threshold_tuning': False,
                                 'other_datasets_changed': False, 'generation_configuration_historical_equivalence': 'UNVERIFIED'}})
    new['datasets']['2wikimultihopqa']['methods'] = {m: {'F1': round(float(np.mean(v) * 100), 2),
                                                        'F1_unrounded': float(np.mean(v) * 100), 'n': len(v)} for m, v in scores.items()}
    assert new['datasets']['2wikimultihopqa']['local_n'] == sum(r['route'] == 'Qwen3.5-9B' for r in rows)
    stats = copy.deepcopy(old_stats)
    stats.update({'status': new['status'], 'repair_scope': new['repair_scope'],
                  'conditional_scope': 'Question-paired uncertainty conditional on retained and four new repaired outputs; not API rerun variability or model-identity verification.'})
    stats['datasets']['2wikimultihopqa'] = {
        'CALIPER_vs_Qwen3.5-9B': paired(scores['CALIPER'], scores['Qwen3.5-9B'], 42),
        'CALIPER_vs_CALIPER+clf': paired(scores['CALIPER'], scores['CALIPER+clf'], 43),
        'CALIPER_vs_Qwen3.7-plus': paired(scores['CALIPER'], scores['Qwen3.7-plus'], 44),
    }
    for dataset in ('hotpotqa', 'musique'):
        assert new['datasets'][dataset] == old['datasets'][dataset]
        assert stats['datasets'][dataset] == old_stats['datasets'][dataset]
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'table3_2wikimultihopqa_paired_scores.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    inputs = [CURRENT / 'table3_current_f1.json', CURRENT / 'table3_paired_bootstrap.json', current_rows_path,
              BASE / 'table-result/table3_full_unified_metrics.json', GENERATION / 'GENERATION_RECEIPT.json',
              GENERATION / 'replacement_predictions.json', GENERATION / 'run_recipe.json',
              ANNOTATION / 'test_supporting_facts_corrected.json', ANNOTATION / 'table7_recall_rescore.json',
              Path(__file__).resolve(), BASE / 'scripts/audit_table3_case_20260930.py', BASE / 'scripts/finalize_table3_case_20260930.py']
    new['input_sha256'].update({str(p): sha(p) for p in inputs})
    write(args.output / 'table3_current_f1.json', new)
    write(args.output / 'table3_paired_bootstrap.json', stats)
    write(args.output / 'four_question_changes.json', changed)
    write(args.output / 'VALIDATION_RECEIPT.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS_FOUR_QUESTION_OVERLAY_AND_PAIRED_RESCORE_NOT_YET_ADOPTED',
        'command': sys.argv, 'n': 12576, 'repaired_question_n': 4,
        'changed_fields_in_paired_rows': ['F1'], 'question_selection_changed': False,
        'classifier_probability_route_threshold_changed': False,
        'original_sources_unchanged': True, 'other_datasets_and_tables_unchanged': True,
        'bootstrap_n': 10000, 'bootstrap_seeds': [42, 43, 44],
        'historical_generation_recipe_equivalence': 'NOT_CLAIMED',
        'input_sha256': new['input_sha256'],
        'output_sha256': {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
    })
    print('TABLE3_BEFORE', json.dumps(old['datasets']['2wikimultihopqa']['methods']), flush=True)
    print('TABLE3_AFTER', json.dumps(new['datasets']['2wikimultihopqa']['methods']), flush=True)
    print('PAIRED_STATISTICS', json.dumps(stats['datasets']['2wikimultihopqa']), flush=True)
    print('FOUR_REPLACEMENTS', json.dumps(changed, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
