"""Owned single-GPU server; sequential datasets, concurrency four per method."""
import json, os, signal, subprocess, time, urllib.request
from pathlib import Path
ROOT=Path('/mnt/data/caliper/code/CALIPER')
RUN=ROOT/'latency_runs/generation_refresh_20260922_v1'
PY='/home/gpt/anaconda3/envs/caliper_new/bin/python'
VPY='/home/gpt/anaconda3/envs/qwen35/bin/python'

def main():
    RUN.mkdir(parents=True,exist_ok=False)
    status={'status':'STARTING','started_at':time.time(),'runs':[]}
    def save(): (RUN/'process_status.json').write_text(json.dumps(status,indent=2))
    def stop(p):
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    cmd=[VPY,'-m','vllm.entrypoints.openai.api_server','--model','/mnt/data/caliper/models/Qwen3.5-9B','--port','6020','--host','127.0.0.1','--trust-remote-code','--max-model-len','8192']
    with (RUN/'vllm.log').open('x') as f:server=subprocess.Popen(cmd,cwd=ROOT,env=os.environ|{'CUDA_VISIBLE_DEVICES':'0'},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    status.update(server_pid=server.pid,server_command=cmd);save()
    try:
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError('server_startup_failed')
            try:
                with urllib.request.urlopen('http://127.0.0.1:6020/v1/models',timeout=3):break
            except Exception:time.sleep(3)
        else:raise TimeoutError('server_startup_timeout')
        for dataset in ('hotpotqa','musique'):
            command=[PY,'-u',str(ROOT/'work/repair/supplement_table5_generation_20260922.py'),dataset]
            record={'dataset':dataset,'command':command,'started_at':time.time(),'timeout_seconds':2700}
            status['runs'].append(record);status['status']='RUNNING';save()
            with (RUN/(dataset+'.log')).open('x') as f:
                child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                record['pid']=child.pid;save()
                try:code=child.wait(timeout=2700)
                except subprocess.TimeoutExpired:stop(child);raise
            record.update(exit_code=code,ended_at=time.time());save()
            if code:raise RuntimeError(dataset+'_failed')
        status['status']='MEASUREMENTS_COMPLETED_PENDING_VALIDATION'
    except Exception as e:status.update(status='STOPPED_FOR_DIAGNOSIS',error=type(e).__name__);raise
    finally:stop(server);status['ended_at']=time.time();save()
if __name__=='__main__':main()
