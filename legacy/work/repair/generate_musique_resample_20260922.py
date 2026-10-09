"""Fresh generation round for the sealed MuSiQue sample; no frozen writes."""
import concurrent.futures,hashlib,json,threading,time
from pathlib import Path
from supplement_table5_generation_20260922 import ROOT,METHODS,historical_functions,assemble

RUN=ROOT/'latency_runs/musique_resample_seed20260922_v1'
RETR=RUN/'retrieval_live_current/musique'
DEST=RUN/'generation'

def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def records(name):return {r['index']:r for r in map(json.loads,(RETR/f'{name}.jsonl').read_text().splitlines())}
def schedule():
    batches=[]
    for block in range(10):
        offset=block%4
        order=METHODS[offset:]+METHODS[:offset]
        for method in order:batches.append({'block':block,'method':method,'indices':list(range(block*20,(block+1)*20))})
    return batches

def main():
    DEST.mkdir(exist_ok=False)
    selection=read(RUN/'SEALED_SAMPLE.json')
    assert selection['status']=='SEALED_BEFORE_MEASUREMENT' and len(selection['selected'])==200
    fixed,core,clf=records('fixed_reference'),records('routed_core'),records('routed_clf')
    assert all(set(rows)==set(range(200)) for rows in (fixed,core,clf))
    items={m:{} for m in METHODS}
    for i,selected in enumerate(selection['selected']):
        assert selected['question']==fixed[i]['question']==core[i]['question']==clf[i]['question']
        branch=clf[i]['classifier_branch']
        for method in METHODS:
            record=(core[i] if method=='caliper' else clf[i] if method=='caliper_clf' and branch=='collaborative' else fixed[i])
            items[method][i]={'index':i,'source_index':selected['source_index'],'question':selected['question'],
                'new_context':record['context'],'expected_branch':branch if method=='caliper_clf' else None,
                'context_sha256':hashlib.sha256(json.dumps(record['context'],ensure_ascii=False).encode()).hexdigest(),
                'retrieval_condition':record['condition'],'retrieval_record':str(RETR/f'{record["condition"]}.jsonl')}
    ns,cn=historical_functions('musique')
    functions={'baseline':ns['base'],'caliper':ns['caliper'],'caliper_clf':ns['caliper_clf_actual'],'cloud_deep':cn['call']}
    state=threading.local()
    def instrument(client,label):
        original=client.chat.completions.create
        def call(**kwargs):
            started=time.perf_counter();response=original(**kwargs)
            state.calls.append({'provider':label,'seconds':time.perf_counter()-started,
                'request':{k:v for k,v in kwargs.items() if k not in ('api_key','headers')},
                'response_content':response.choices[0].message.content or '',
                'finish_reason':response.choices[0].finish_reason,'usage':response.usage.model_dump() if response.usage else {},'model':response.model})
            return response
        client.chat.completions.create=call
    instrument(ns['local'],'local');instrument(ns['cloud'],'cloud_guidance');instrument(cn['client'],'cloud_baseline')
    evidence=[RUN/'SEALED_SAMPLE.json',RETR/'fixed_reference.jsonl',RETR/'routed_core.jsonl',RETR/'routed_clf.jsonl',Path(__file__),Path(__file__).with_name('supplement_table5_generation_20260922.py'),ROOT/'scripts/measure_table4_final.py',ROOT/'scripts/measure_cloud_deep.py']
    manifest={'status':'RUNNING_CANDIDATE_NOT_FROZEN','created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'dataset':'musique','methods':METHODS,'n':200,'warmup_per_method':20,'workers_global':4,
        'selection_seed':selection['seed'],'schedule':schedule(),'input_sha256':{str(p):sha(p) for p in evidence},
        'local_generation':{'temperature':.1,'baseline_tokens':128,'caliper_tokens':[96,32],'thinking':False,'request_seed':None},
        'cloud_generation':{'temperature':.2,'guidance_tokens':64,'baseline_tokens':[256,64],'thinking_budget':256,'request_seed':None},
        'error_policy':'Preserve failures and slow successes; no replacement, retry, imputation, or outlier deletion.',
        'scope':'All generation timings fresh; composed later with this sample fresh retrieval-path timings; not integrated live measurement.'}
    dump(DEST/'manifest.json',manifest)
    def one(method,index,phase,block):
        item=items[method][index];state.calls=[];started=time.perf_counter()
        try:
            result=functions[method](item)
            if result.get('error'):result['error']='request_failed'
            if not result.get('error') and method=='caliper_clf' and result['route']!=item['expected_branch']:result['error']='classifier_branch_mismatch'
            if not result.get('error') and any(not c['response_content'].strip() for c in state.calls):result['error']='empty_response'
        except Exception as exc:
            result={'seconds':time.perf_counter()-started,'error':type(exc).__name__}
        return dict(item,**result,method=method,phase=phase,block=block,calls=state.calls,measured_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    errors=[];completed=0;started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for method in METHODS:
            with (DEST/f'{method}_warmup.jsonl').open('x') as out:
                for row in executor.map(lambda i:one(method,i,'warmup',-1),range(20)):
                    out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                    if row.get('error'):errors.append((method,row['index'],'warmup',row['error']))
            print('WARMUP_DONE',method,'errors',len(errors),flush=True)
        for batch in schedule():
            method=batch['method']
            with (DEST/f'{method}_measured.jsonl').open('a') as out:
                futures=[executor.submit(one,method,i,'measured',batch['block']) for i in batch['indices']]
                for future in concurrent.futures.as_completed(futures):
                    row=future.result();out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();completed+=1
                    if row.get('error'):errors.append((method,row['index'],'measured',row['error']))
            dump(DEST/'progress.json',{'completed':completed,'target':800,'errors':errors,'last_batch':batch,'elapsed_seconds':time.time()-started})
            print('BATCH_DONE',completed,'/800',method,batch['block'],'errors',len(errors),flush=True)
    manifest.update(status='MEASUREMENTS_COMPLETED_PENDING_VALIDATION' if not errors else 'COMPLETED_WITH_ERRORS',errors=errors,ended_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    dump(DEST/'manifest.json',manifest);print('COMPLETED',completed,'errors',len(errors),flush=True)

if __name__=='__main__':main()
