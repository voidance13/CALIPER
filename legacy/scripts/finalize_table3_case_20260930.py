"""Apply the approved two-row Table 3 exclusion to every compared method.

Reads existing branch predictions and classifier probabilities only. Preserves
historical result files, model, thresholds and every other paper table.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_table3_case_20260930 import BASE, NEW, THRESHOLDS, canonical, first, f1, gold, load, pred, sha


def paired(left, right, seed, samples=10000):
    delta = np.asarray(left) - np.asarray(right)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples)
    for start in range(0, samples, 100):
        count = min(100, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        bootstrap[start:start + count] = delta[indices].mean(axis=1)
    below = (np.count_nonzero(bootstrap <= 0) + 1) / (samples + 1)
    above = (np.count_nonzero(bootstrap >= 0) + 1) / (samples + 1)
    return {'n': len(delta), 'left_F1': round(float(np.mean(left) * 100), 2),
            'right_F1': round(float(np.mean(right) * 100), 2),
            'delta_F1': round(float(delta.mean() * 100), 2),
            'delta_CI95': [round(float(x * 100), 2) for x in np.percentile(bootstrap, (2.5, 97.5))],
            'p_two_sided': round(min(1.0, 2 * min(below, above)), 6),
            'bootstrap_samples': samples, 'seed': seed}


def questions(path):
    return {canonical(line.split(' ', 1)[1]) for line in path.read_text().splitlines() if line.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = BASE / 'classifier_analysis/table3_case_repair_20260930'
    audit = load(source / 'table3_classifier_audit.json')
    old_path = BASE / 'table-result/table3_full_unified_metrics.json'
    old = load(old_path)
    assert sha(old_path) == audit['old_table3_sha256']
    clf_path = NEW / '9b_clf_overlap_free.bin'
    assert sha(clf_path) == audit['model_sha256']
    train_path, fitvalid_path = [NEW / f'9b_{name}_overlap_free.txt' for name in ('train', 'valid')]
    train, fitvalid = questions(train_path), questions(fitvalid_path)
    threshold_pools = {d: questions(BASE / f'query_clf/train_new/l1o_{d}_valid.txt') - train
                       for d in THRESHOLDS}
    result = {'status': 'USER_APPROVED_TABLE3_REPAIR_PENDING_PAPER_ARCHIVE_SYNC',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'protocol': 'Reuse frozen fully-relevant-evidence answers; first occurrence per exact question; exclude fitting-overlap questions identically for all four methods; select local when p_easy >= the frozen dataset threshold, otherwise CALIPER.',
              'metric_scope': 'F1 only, matching the printed Table 3; no newly computed SBERT/Hit/P/R claims.',
              'no_generation_cloud_training_or_threshold_tuning': True,
              'classifier_model_sha256': sha(clf_path), 'thresholds': THRESHOLDS,
              'old_table3_sha256': sha(old_path), 'datasets': {}, 'input_sha256': {}}
    statistics = {'status': result['status'], 'definition': 'Question-paired token F1, 10,000 resamples, percentile 95% CI and two-sided finite-sample +1 p-value; original comparison seeds retained.', 'datasets': {}}
    for dataset, threshold in THRESHOLDS.items():
        test_path = BASE / f'data/{dataset}/test.json'
        tests = first(load(test_path))
        removed = [q for q in tests if canonical(q) in train | fitvalid]
        retained = [q for q in tests if q not in removed]
        assert not any(canonical(q) in threshold_pools[dataset] for q in retained)
        assert len(retained) == {'hotpotqa': 7405, 'musique': 2410, '2wikimultihopqa': 12576}[dataset]
        audit_path = source / f'table3_{dataset}_new_classifier_audit.jsonl'
        probability_rows = {r['question']: r for r in [json.loads(l) for l in audit_path.read_text().splitlines()]}
        scores = {}
        sources = {}
        for method, meta in old['datasets'][dataset]['sources'].items():
            path = BASE / meta['path']
            assert sha(path) == meta['sha256']
            sources[method] = first(load(path))
            scores[method] = [f1(pred(sources[method][q]), gold(tests[q])) for q in retained]
            result['input_sha256'][str(path)] = sha(path)
        scores['CALIPER+clf'] = []
        output_rows = []
        for q in retained:
            p = probability_rows[q]['p_easy']
            assert probability_rows[q]['threshold'] == threshold
            branch = 'Qwen3.5-9B' if p >= threshold else 'CALIPER'
            score = f1(pred(sources[branch][q]), gold(tests[q]))
            assert abs(score - probability_rows[q]['selected_F1']) < 1e-12
            scores['CALIPER+clf'].append(score)
            output_rows.append({'question': q, 'gold': gold(tests[q]), 'p_easy': p,
                                'threshold': threshold, 'route': branch,
                                'F1': {m: f1(pred(sources[m][q]), gold(tests[q])) for m in sources} | {'CALIPER+clf': score}})
        with (args.output / f'table3_{dataset}_paired_scores.jsonl').open('w') as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        result['datasets'][dataset] = {
            'original_n': len(tests), 'n': len(retained), 'excluded_questions': removed,
            'methods': {m: {'F1': round(float(np.mean(v) * 100), 2),
                            'F1_unrounded': float(np.mean(v) * 100), 'n': len(v)} for m, v in scores.items()},
            'local_n': sum(r['route'] == 'Qwen3.5-9B' for r in output_rows),
            'overlap_check': {'train_n': 0, 'fitting_validation_n': 0, 'own_threshold_selection_n': 0,
                              'threshold_selection_by_other_datasets': {d: sum(canonical(q) in threshold_pools[d] for q in retained)
                                                                        for d in THRESHOLDS if d != dataset}}}
        statistics['datasets'][dataset] = {
            'CALIPER_vs_Qwen3.5-9B': paired(scores['CALIPER'], scores['Qwen3.5-9B'], 42),
            'CALIPER_vs_CALIPER+clf': paired(scores['CALIPER'], scores['CALIPER+clf'], 43),
            'CALIPER_vs_Qwen3.7-plus': paired(scores['CALIPER'], scores['Qwen3.7-plus'], 44)}
        for path in [test_path, audit_path]:
            result['input_sha256'][str(path)] = sha(path)
        print(dataset, json.dumps(result['datasets'][dataset]), flush=True)
    for path in [train_path, fitvalid_path] + [BASE / f'query_clf/train_new/l1o_{d}_valid.txt' for d in THRESHOLDS]:
        result['input_sha256'][str(path)] = sha(path)
    case = load(source / 'selected_misrouting_case.json')
    for path, expected in case['sources_sha256'].items():
        assert sha(Path(path)) == expected
    assert case['model_sha256'] == result['classifier_model_sha256']
    assert case['p_easy'] >= case['threshold'] == THRESHOLDS['musique']
    for name, value in [('table3_current_f1.json', result), ('table3_paired_bootstrap.json', statistics), ('selected_misrouting_case.json', case)]:
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print('PAIRED_STATISTICS', json.dumps(statistics['datasets']), flush=True)


if __name__ == '__main__':
    main()
