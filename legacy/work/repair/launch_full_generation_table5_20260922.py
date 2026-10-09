import json,os,signal,subprocess,time,urllib.request
from pathlib import Path
from full_generation_table5_20260922 import ROOT,RUN
PY='/home/gpt/anaconda3/envs/caliper_new/bin/python'
VPY='/home/gpt/anaconda3/envs/qwen35/bin/python'

def main():
    RUN.mkdir(parents=True,exist_ok=False)
    status={'status':'STARTING','started_at':time.time()}
    def save():(RUN/'process_status.json').write_text(json.dumps(status,indent=2))
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
        command=[PY,'-u',str(ROOT/'work/repair/full_generation_table5_20260922.py')]
        with (RUN/'generation.log').open('x') as f:child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        status.update(status='RUNNING',command=command,pid=child.pid,timeout_seconds=10800);save()
        try:code=child.wait(timeout=10800)
        except subprocess.TimeoutExpired:stop(child);raise
        status.update(exit_code=code,status='MEASUREMENTS_COMPLETED_PENDING_VALIDATION' if code==0 else 'STOPPED_FOR_DIAGNOSIS')
    except Exception as e:status.update(status='STOPPED_FOR_DIAGNOSIS',error=type(e).__name__);raise
    finally:stop(server);status['ended_at']=time.time();save()
if __name__=='__main__':main()
