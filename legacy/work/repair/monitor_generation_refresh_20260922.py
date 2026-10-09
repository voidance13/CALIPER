import json
from pathlib import Path
root=Path('/mnt/data/caliper/code/CALIPER/latency_runs/generation_refresh_20260922_v1')
status=json.loads((root/'process_status.json').read_text())
print('PROCESS',status['status'])
for d in ('hotpotqa','musique'):
    for p in sorted((root/d).glob('*_measured.jsonl')):
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        good=[r for r in rows if not r.get('error')]
        controls=[r for r in good if r['role']=='control']
        print(d,p.stem,'n',len(rows),'errors',sum(bool(r.get('error')) for r in rows),
              'mean',round(sum(r['seconds'] for r in good)/len(good),3) if good else None,
              'control_ratio',round(sum(r['seconds'] for r in controls)/sum(r['old_generation_seconds'] for r in controls),3) if controls else None)
    log=root/(d+'.log')
    if log.exists():print('LATEST',d,log.read_text().splitlines()[-1])
