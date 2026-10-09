"""Freeze the user-approved V4 artifacts without rewriting raw results."""
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

RUN = Path('/mnt/data/caliper/code/CALIPER/router_e2e_runs/v4_v3aligned_20260921')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

metrics = json.loads((RUN / 'aligned_v4_metrics_bootstrap.json').read_text())
expected = {'hotpotqa': (7405, 67.00), 'musique': (2403, 40.56), '2wikimultihopqa': (12576, 62.99)}
files = {}
for dataset, (count, f1) in expected.items():
    record = metrics['datasets'][dataset]
    assert record['metrics']['n'] == count and record['metrics']['F1'] == f1
    assert record['empty_predictions'] == 0
    for name, digest in record['evidence'].items():
        assert sha(Path(name)) == digest, name
    for suffix in ('v4_predictions.json', 'paired_scores.json'):
        path = RUN / f'{dataset}_{suffix}'
        rows = json.loads(path.read_text())
        assert len(rows) == count
        files[path.name] = sha(path)
for name in ('manifest.json', 'aligned_v4_metrics_bootstrap.json', 'GENERATION_COMPLETE.json', 'manuscript_v4_frozen.tex'):
    files[name] = sha(RUN / name)
receipt = {
    'version': 'v4_v3aligned_20260921',
    'status': 'FROZEN',
    'scope': 'Table 4 V4 only; not a freeze of the whole manuscript or Table 8',
    'frozen_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'authority': 'User approved freezing V4 and replacing manuscript values and interpretations.',
    'verification': 'PASS: retained evidence hashes, complete prediction and score counts, expected F1, no empty predictions',
    'f1': {key: value[1] for key, value in expected.items()},
    'supersedes_v4_f1': {'hotpotqa': 63.47, 'musique': 39.00, '2wikimultihopqa': 59.11},
    'statistics_scope': 'Existing V1/V3 references only; recompute comparisons when reference predictions change.',
    'raw_status_note': 'Original candidate/preflight status strings are retained as historical metadata; this receipt records adoption.',
    'archive_status': 'This run directory is the frozen V4 evidence bundle. Previous full companion archive does not contain this update.',
    'files_sha256': files,
}
target = RUN / 'FREEZE_RECEIPT.json'
with target.open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt, indent=2))
