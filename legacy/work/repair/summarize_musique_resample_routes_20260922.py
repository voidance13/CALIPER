import json
from collections import Counter
from pathlib import Path
p=Path('/mnt/data/caliper/code/CALIPER/latency_runs/musique_resample_seed20260922_v1/generation/caliper_clf_measured.jsonl')
rows=list(map(json.loads,p.read_text().splitlines()))
print(json.dumps({'n':len(rows),'branches':Counter(r.get('route') for r in rows),'errors':Counter(r.get('error') for r in rows)},default=dict))
