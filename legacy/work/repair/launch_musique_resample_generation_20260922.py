import json,os,signal,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path('/mnt/data/caliper/code/CALIPER')
RUN=ROOT/'latency_runs/musique_resample_seed20260922_v1'
PY='/home/gpt/anaconda3/envs/caliper_new/bin/python';VPY='/home/gpt/anaconda3/envs/qwen35/bin/python'
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def stop(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
def main():
    assert (RUN/'retrieval_live_current/musique/summary.json').exists()
    status={'status':'STARTING','started_at':time.time()};dump(RUN/'generation_process_status.json',status)
    command=[VPY,'-m','vllm.entrypoints.openai.api_server','--model','/mnt/data/caliper/models/Qwen3.5-9B','--port','6020','--host','127.0.0.1','--trust-remote-code','--max-model-len','8192']
    with (RUN/'generation_vllm.log').open('x') as log:
        server=subprocess.Popen(command,cwd=ROOT,env=os.environ|{'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError('server_startup_failed')
            try:
                with urllib.request.urlopen('http://127.0.0.1:6020/v1/models',timeout=3):break
            except Exception:time.sleep(3)
        else:raise TimeoutError('server_startup_timeout')
        child_command=[PY,'-u',str(ROOT/'work/repair/generate_musique_resample_20260922.py')]
        with (RUN/'generation.log').open('x') as log:
            child=subprocess.Popen(child_command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status.update(status='RUNNING',server_pid=server.pid,pid=child.pid,command=child_command,timeout_seconds=7200);dump(RUN/'generation_process_status.json',status)
        try:code=child.wait(timeout=7200)
        except subprocess.TimeoutExpired:stop(child);raise
        status.update(exit_code=code,status='MEASUREMENTS_COMPLETED_PENDING_VALIDATION' if code==0 else 'STOPPED_FOR_DIAGNOSIS')
    except Exception as exc:
        status.update(status='STOPPED_FOR_DIAGNOSIS',error_type=type(exc).__name__);raise
    finally:
        stop(server);status['ended_at']=time.time();dump(RUN/'generation_process_status.json',status)
if __name__=='__main__':main()
