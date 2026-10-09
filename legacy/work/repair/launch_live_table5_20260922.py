"""Sequential datasets, identical hardware, bounded subprocesses, no retries."""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path('/mnt/data/caliper/code/CALIPER')
RUN=ROOT/'latency_runs/live_table5_20260922_v2'
PY='/home/gpt/anaconda3/envs/caliper_new/bin/python'
VPY='/home/gpt/anaconda3/envs/qwen35/bin/python'


def main():
    RUN.mkdir(parents=True,exist_ok=False)
    status={'status':'STARTING','started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'runs':[]}
    def save():
        (RUN/'process_status.json').write_text(json.dumps(status,indent=2))
    server_cmd=[VPY,'-m','vllm.entrypoints.openai.api_server','--model','/mnt/data/caliper/models/Qwen3.5-9B',
        '--port','6020','--host','127.0.0.1','--trust-remote-code','--max-model-len','8192']
    with (RUN/'vllm.log').open('x') as log:
        server=subprocess.Popen(server_cmd,cwd=ROOT,env=os.environ|{'CUDA_VISIBLE_DEVICES':'0'},
                                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status.update(server_pid=server.pid,server_command=server_cmd); save()
    try:
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if server.poll() is not None: raise RuntimeError('vLLM exited during startup')
            try:
                with urllib.request.urlopen('http://127.0.0.1:6020/v1/models',timeout=3): break
            except Exception: time.sleep(3)
        else: raise TimeoutError('vLLM startup timeout')
        for phase,limit,warmup in [('smoke',2,0),('full',200,20)]:
            for dataset in ('hotpotqa','musique','2wikimultihopqa'):
                command=[PY,'-u',str(ROOT/'work/repair/live_table5_20260922.py'),
                    '--dataset',dataset,'--output',str(RUN/phase),'--limit',str(limit),'--warmup',str(warmup)]
                record={'phase':phase,'dataset':dataset,'command':command,'started_at':time.time(),'timeout_seconds':1800}
                status['runs'].append(record); status['status']='RUNNING'; save()
                with (RUN/f'{phase}_{dataset}.log').open('x') as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    record['pid']=child.pid; save()
                    try: code=child.wait(timeout=1800)
                    except subprocess.TimeoutExpired:
                        print('HARD_TIMEOUT: terminating owned benchmark process',flush=True)
                        child.terminate()
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                        raise
                record.update(exit_code=code,ended_at=time.time()); save()
                if code: raise RuntimeError(f'{phase}/{dataset} failed; no retries')
                print('RUN_DONE',phase,dataset,flush=True)
        status['status']='COMPLETED_CANDIDATE_NOT_FROZEN'
    except Exception as e:
        status.update(status='STOPPED_REQUIRES_REVIEW',error=str(e)); raise
    finally:
        status['ended_at']=time.time(); save()
        # This server is owned by this launcher; normal lifecycle cleanup only.
        server.terminate()
        try: server.wait(timeout=30)
        except subprocess.TimeoutExpired: server.kill(); server.wait()


if __name__=='__main__':
    main()
