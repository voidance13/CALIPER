import json
from full_generation_table5_20260922 import RUN
status=json.loads((RUN/'process_status.json').read_text());print('PROCESS',status['status'])
written=sum(len(p.read_text().splitlines()) for p in (RUN/'measurements').glob('*_measured.jsonl'))
print('WRITTEN',written,'/2400')
progress=RUN/'progress.json'
if progress.exists():
    p=json.loads(progress.read_text());print('PROGRESS',p['completed'],'/2400','errors',len(p['errors']),'elapsed_minutes',round(p['elapsed_seconds']/60,1));print('LAST',p['last_block']['dataset'],p['last_block']['method'],p['last_block']['block'])
log=RUN/'generation.log'
if log.exists():print('LATEST',log.read_text().splitlines()[-1:])
