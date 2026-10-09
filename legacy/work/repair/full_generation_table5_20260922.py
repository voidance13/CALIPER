"""One contemporaneous generation round, interleaved blocks, frozen retrieval inputs."""
import concurrent.futures,json,random,threading,time
from pathlib import Path
from supplement_table5_generation_20260922 import ROOT,OLD,METHODS,prepare,historical_functions,sha,dump,assemble
RUN=ROOT/'latency_runs/full_generation_table5_20260922_v1'
DATASETS=('hotpotqa','musique','2wikimultihopqa')

def schedule():
    result=[]
    for block in range(10):
        ds=DATASETS[block%3:]+DATASETS[:block%3]
        for d in ds:
            offset=(block+DATASETS.index(d))%4
            methods=METHODS[offset:]+METHODS[:offset]
            for m in methods:result.append({'block':block,'dataset':d,'method':m,'indices':list(range(block*20,(block+1)*20))})
    return result

def main():
    dest=RUN/'measurements';dest.mkdir(exist_ok=False)
    state=threading.local();all_items={};funcs={};inputs={};cohorts={}
    for d in DATASETS:
        rows,items,paths=prepare(d);all_items[d]=items
        cohorts[d]={m:[r['index'] for r in rows if r['method']==m and r['error'] is None] for m in METHODS}
        ns,cn=historical_functions(d)
        funcs[d]={'baseline':ns['base'],'caliper':ns['caliper'],'caliper_clf':ns['caliper_clf_actual'],'cloud_deep':cn['call']}
        def instrument(client,label):
            original=client.chat.completions.create
            def call(**kwargs):
                started=time.perf_counter()
                result=original(**kwargs)
                state.calls.append({'provider':label,'seconds':time.perf_counter()-started,
                    'request':{k:v for k,v in kwargs.items() if k not in ('api_key','headers')},
                    'response_content':result.choices[0].message.content or '',
                    'finish_reason':result.choices[0].finish_reason,
                    'usage':result.usage.model_dump() if result.usage else {},'model':result.model})
                return result
            client.chat.completions.create=call
        instrument(ns['local'],'local');instrument(ns['cloud'],'cloud_guidance');instrument(cn['client'],'cloud_baseline')
        for p in [*paths.values(),OLD/f'{d}_paired_composition.json']:inputs[str(p)]=sha(p)
    for p in [Path(__file__),Path(__file__).with_name('supplement_table5_generation_20260922.py'),ROOT/'scripts/measure_table4_final.py',ROOT/'scripts/measure_cloud_deep.py']:
        inputs[str(p)]=sha(p)
    manifest={'status':'RUNNING_CANDIDATE_NOT_FROZEN','created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'datasets':DATASETS,'methods':METHODS,'questions_per_dataset':200,'workers_global':4,'warmup_per_dataset_method':20,
        'measurement_schedule':schedule(),'primary_valid_indices':cohorts,'input_sha256':inputs,
        'local_generation':{'temperature':.1,'baseline_tokens':128,'caliper_tokens':[96,32],'thinking':False,'request_seed':None},
        'cloud_generation':{'temperature':.2,'guidance_tokens':64,'baseline_tokens':[256,64],'thinking_budget':256,'request_seed':None},
        'scope':'All generation timings new in this round; previously measured per-question retrieval durations reused. Not an integrated end-to-end measurement or Table 2 generation configuration.',
        'route_policy':'Frozen retrieval records including MuSiQue indices 72/86 replay, no live-route-equivalence claim',
        'error_policy':'Preserve all attempts and slow successes; no outlier deletion or best-of-repeat selection. Failed batches require explicit recorded repair, never silently replaced.',
        'sampling':'Original seed-42 question list; primary cohorts preserve historical exclusions; all-200 sensitivity separately reported if complete.'}
    dump(RUN/'manifest.json',manifest)
    def one(d,m,i,phase,block):
        item=all_items[d][m][i];state.calls=[]
        result=funcs[d][m](item)
        if result.get('error'):result['error']='request_failed'
        if not result.get('error') and m=='caliper_clf' and result['route']!=item['expected_branch']:result['error']='classifier_branch_mismatch'
        if not result.get('error') and any(not c['response_content'].strip() for c in state.calls):result['error']='empty_response'
        return dict(item,**result,dataset=d,method=m,phase=phase,block=block,calls=state.calls,
            primary_included=i in cohorts[d][m],measured_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    completed=0;errors=[];started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for d in DATASETS:
            for m in METHODS:
                with (dest/f'{d}_{m}_warmup.jsonl').open('x') as f:
                    for r in ex.map(lambda i:one(d,m,i,'warmup',-1),range(20)):
                        f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
                        if r.get('error'):errors.append((d,m,r['index'],'warmup',r['error']))
                print('WARMUP_DONE',d,m,flush=True)
        for s in schedule():
            d,m,block=s['dataset'],s['method'],s['block']
            with (dest/f'{d}_{m}_measured.jsonl').open('a') as f:
                futures=[ex.submit(one,d,m,i,'measured',block) for i in s['indices']]
                for future in concurrent.futures.as_completed(futures):
                    r=future.result();f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();completed+=1
                    if r.get('error'):errors.append((d,m,r['index'],'measured',r['error']))
            dump(RUN/'progress.json',{'completed':completed,'target':2400,'errors':errors,'last_block':s,'elapsed_seconds':time.time()-started})
            print('BATCH_DONE',completed,'/2400',d,m,block,'errors',len(errors),flush=True)
    manifest.update(status='MEASUREMENTS_COMPLETED_PENDING_VALIDATION' if not errors else 'COMPLETED_WITH_ERRORS',errors=errors,ended_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    dump(RUN/'manifest.json',manifest)
    print('COMPLETED',completed,'errors',len(errors),flush=True)

if __name__=='__main__':main()
