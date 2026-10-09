import json,os,signal,subprocess,time,urllib.request
from pathlib import Path
from supplement_table5_generation_20260922 import ROOT,prepare,historical_functions,read,dump,sha
SOURCE=ROOT/'latency_runs/full_generation_table5_20260922_v1'
DEST=ROOT/'latency_runs'/('musique_tail_diagnostic_'+time.strftime('%Y%m%d_%H%M%S'))

def main():
    DEST.mkdir(exist_ok=False)
    source=SOURCE/'measurements/musique_caliper_clf_measured.jsonl'
    old=next(r for r in map(json.loads,source.read_text().splitlines()) if r['index']==101)
    oldhash=sha(source)
    _,items,_=prepare('musique');item=items['caliper_clf'][101]
    assert item['question']==old['question'] and item['new_context']==old['new_context']
    command=['/home/gpt/anaconda3/envs/qwen35/bin/python','-m','vllm.entrypoints.openai.api_server','--model','/mnt/data/caliper/models/Qwen3.5-9B','--port','6020','--host','127.0.0.1','--trust-remote-code','--max-model-len','8192']
    status={'status':'STARTING','source':str(source),'source_sha256':oldhash,'single_question_diagnostic':True,'concurrency':1,'primary_result_replaced':False}
    dump(DEST/'status.json',status);print(str(DEST),flush=True)
    with (DEST/'server.log').open('x') as log:
        server=subprocess.Popen(command,cwd=ROOT,env=os.environ|{'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        deadline=time.monotonic()+600
        while True:
            assert server.poll() is None,'server stopped'
            try:
                with urllib.request.urlopen('http://127.0.0.1:6020/v1/models',timeout=3):break
            except Exception:
                if time.monotonic()>deadline:raise TimeoutError('startup')
                time.sleep(3)
        ns,_=historical_functions('musique')
        # One local-only warmup; no additional cloud questions or selected repeats.
        warm=ns['base'](item)
        if warm.get('error'):raise RuntimeError('warmup_failed')
        dump(DEST/'local_warmup.json',warm)
        calls=[]
        def instrument(client,label):
            original=client.chat.completions.create
            def call(**kwargs):
                started=time.perf_counter();response=original(**kwargs)
                calls.append({'provider':label,'seconds':time.perf_counter()-started,'request':{k:v for k,v in kwargs.items() if k not in ('api_key','headers')},'response_content':response.choices[0].message.content or '', 'model':response.model})
                return response
            client.chat.completions.create=call
        instrument(ns['local'],'local');instrument(ns['cloud'],'cloud_guidance')
        status['status']='MEASURING';dump(DEST/'status.json',status)
        result=ns['caliper_clf_actual'](item)
        if result.get('error'):result['error']='request_failed'
        dump(DEST/'attempt.json',dict(item,**result,calls=calls,measured_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        assert not result.get('error'),'request_failed'
        assert result['route']==item['expected_branch']
        assert [c['provider'] for c in calls]==[c['provider'] for c in old['calls']]
        assert calls[0]['request']==old['calls'][0]['request'],'cloud guidance request changed'
        for a,b in zip(calls,old['calls']):
            assert {k:v for k,v in a['request'].items() if k!='messages'}=={k:v for k,v in b['request'].items() if k!='messages'}
            assert a['response_content'].strip()
        composed=next(r for r in read(SOURCE/'analysis/musique_paired_composition.json') if r['method']=='caliper_clf' and r['index']==101)
        assert sha(source)==oldhash
        status.update(status='COMPLETED_DIAGNOSTIC_NOT_REPLACEMENT',old_generation_seconds=old['seconds'],new_generation_seconds=result['seconds'],old_estimated_total=composed['seconds'],new_estimated_total=composed['retrieval_seconds']+result['seconds'],calls_seconds=[{'provider':c['provider'],'seconds':c['seconds']} for c in calls],checks='PASS',limitations='Single request after one local warmup, not original concurrency-four workload; fresh cloud guidance can change downstream text. No attribution of original delay cause.')
    except Exception as e:
        status.update(status='FAILED',error_type=type(e).__name__)
        raise
    finally:
        if server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        dump(DEST/'status.json',status)
        print(json.dumps(status),flush=True)

if __name__=='__main__':main()
