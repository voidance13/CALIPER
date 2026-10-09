"""Launch the approved three-dataset timing run, each at concurrency four."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/mnt/data/caliper/code/CALIPER')
output = root / 'retrieval_runs/path_latency_per_question_20260922_v1'
output.mkdir(exist_ok=False)
script = root / 'work/repair/measure_route_latency_20260922.py'
datasets = ('hotpotqa', 'musique', '2wikimultihopqa')
processes, handles, commands = {}, [], {}
started = time.time()
for gpu, dataset in enumerate(datasets):
    command = ['timeout', '1800', sys.executable, '-u', str(script), '--dataset', dataset,
               '--gpu', str(gpu), '--output', str(output)]
    log = (output / f'{dataset}.log').open('x')
    handles.append(log)
    commands[dataset] = command
    processes[dataset] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
        cwd=root, env=dict(os.environ, OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false'))
while True:
    states = {d: p.poll() for d, p in processes.items()}
    status = {'started_at': started, 'elapsed_seconds': time.time() - started,
              'pids': {d: p.pid for d, p in processes.items()}, 'exit_codes': states,
              'commands': commands, 'frozen': False}
    (output / 'process_status.json').write_text(json.dumps(status, indent=2))
    if all(v is not None for v in states.values()):
        break
    time.sleep(15)
for log in handles:
    log.close()
if any(v != 0 for v in states.values()):
    raise SystemExit('One or more datasets failed; no automatic retry. See process_status.json.')
summary = {d: json.loads((output / d / 'summary.json').read_text()) for d in datasets}
(output / 'all_summary.json').write_text(json.dumps(summary, indent=2))
print('COMPLETED_CANDIDATE_NOT_FROZEN', output, flush=True)
