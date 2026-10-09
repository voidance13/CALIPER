"""Selective generation refresh; historical prompt functions retained via AST.

No frozen file writes. Controls are diagnostics, never extra independent questions.
"""
import ast
import concurrent.futures
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('/mnt/data/caliper/code/CALIPER')
RUN=ROOT/'latency_runs/generation_refresh_20260922_v1'
OLD=ROOT/'latency_runs/composed_table5_20260922_v1'
METHODS=('baseline','caliper','caliper_clf','cloud_deep')

def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def selection(records,method):
    valid=[r for r in records if r['method']==method and r['error'] is None]
    changed=[r['index'] for r in valid if not r['old_generation_context_exact_match']]
    stable=[r['index'] for r in valid if r['old_generation_context_exact_match']]
    control=random.Random(42).sample(stable,min(20,len(stable)))
    return changed,control
def assemble(context):return '\n'.join(f'[{i+1}] {s}' for i,s in enumerate(context))

def prepare(dataset):
    rows=read(OLD/f'{dataset}_paired_composition.json')
    v='v2' if dataset=='musique' else 'v1'
    paths={n:ROOT/f'retrieval_runs/path_latency_per_question_20260922_{v}/{dataset}/{n}.jsonl' for n in ('fixed_reference','routed_core','routed_clf')}
    retrieval={n:{r['index']:r for r in map(json.loads,p.read_text().splitlines())} for n,p in paths.items()}
    by={(r['method'],r['index']):r for r in rows}
    items={}
    for method in METHODS:
        items[method]={}
        for i in range(200):
            old=by[method,i]
            kind='routed_core' if method=='caliper' else 'routed_clf' if method=='caliper_clf' and old['classifier_branch']=='collaborative' else 'fixed_reference'
            r=retrieval[kind][i]
            assert r['question']==old['question'] and r['error'] is None
            assert r['matches_old_generation_context']==old['old_generation_context_exact_match']
            items[method][i]={'index':i,'question':r['question'],'new_context':r['context'],'expected_branch':old['classifier_branch'],
                'context_sha256':hashlib.sha256(json.dumps(r['context'],ensure_ascii=False).encode()).hexdigest(),
                'frozen_route_replayed':old['frozen_route_replayed'],'retrieval_record':str(paths[kind]),'old_generation_seconds':old['historical_generation_seconds']}
    return rows,items,paths

def historical_functions(dataset):
    """Load only setup and function definitions, never historical output-writing code."""
    p=ROOT/'scripts/measure_table4_final.py'; tree=ast.parse(p.read_text())
    prefix=[]
    for n in tree.body:
        if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='MODE':break
        prefix.append(n)
    assert any(isinstance(n,ast.FunctionDef) and n.name=='caliper_clf_actual' for n in prefix)
    prior=sys.argv;sys.argv=[str(p),dataset,'6020','supplement']
    ns={'__name__':'historical_functions_only'}
    try: exec(compile(ast.Module(body=prefix,type_ignores=[]),str(p),'exec'),ns)
    finally: sys.argv=prior
    ns['context']=lambda item:assemble(item['new_context'])
    cp=ROOT/'scripts/measure_cloud_deep.py';ct=ast.parse(cp.read_text())
    main=next(n for n in ct.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    defs=[n for n in main.body if isinstance(n,ast.FunctionDef)]
    cn={'time':time,'MODEL':'qwen3.7-plus','args':SimpleNamespace(thinking_budget=256,workers=4),
        'client':ns['OpenAI'](api_key=os.environ['ALIYUN_API_KEY'],base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',timeout=180)}
    exec(compile(ast.Module(body=defs,type_ignores=[]),str(cp),'exec'),cn)
    cn['context']=ns['context']
    return ns,cn

def run(dataset):
    dest=RUN/dataset;dest.mkdir(parents=True,exist_ok=False)
    rows,items,paths=prepare(dataset)
    plans={m:dict(zip(('changed','controls'),selection(rows,m))) for m in METHODS}
    evidence=list(paths.values())+[OLD/f'{dataset}_paired_composition.json',ROOT/'scripts/measure_table4_final.py',ROOT/'scripts/measure_cloud_deep.py',Path(__file__)]
    manifest={'status':'RUNNING_NOT_FROZEN','dataset':dataset,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
      'plans':plans,'workers':4,'warmup_per_method':20,'sample_seed':42,
      'generation_request_seed':'not supplied by historical functions; unchanged',
      'local_temperature':.1,'cloud_temperature':.2,'local_baseline_tokens':128,'caliper_tokens':[96,32],
      'cloud_guidance_tokens':64,'cloud_baseline_tokens':[256,64],'thinking_budget':256,
      'route_policy':'retained measured routes, including disclosed MuSiQue 72/86 frozen-route replay; no claim of live-route equivalence',
      'client_retry_policy':'historical OpenAI SDK default retained; additional retries if any are separate versioned attempts',
      'hashes':{str(p):sha(p) for p in evidence},'not_integrated_retrieval_generation':True}
    dump(dest/'manifest.json',manifest)
    ns,cn=historical_functions(dataset)
    functions={'baseline':ns['base'],'caliper':ns['caliper'],'caliper_clf':ns['caliper_clf_actual'],'cloud_deep':cn['call']}
    print('READY',dataset,{m:{k:len(v) for k,v in p.items()} for m,p in plans.items()},flush=True)
    # Instrument requests without changing their prompts or request parameters.
    import threading
    state=threading.local()
    def instrument(client,label):
        original=client.chat.completions.create
        def call(**kwargs):
            started=time.perf_counter();result=original(**kwargs)
            content=result.choices[0].message.content or ''
            state.calls.append({'provider':label,'seconds':time.perf_counter()-started,
                'request':{k:v for k,v in kwargs.items() if k not in ('api_key','headers')},
                'response_content':content,'finish_reason':result.choices[0].finish_reason,
                'usage':result.usage.model_dump() if result.usage else {},'model':result.model})
            return result
        client.chat.completions.create=call
    instrument(ns['local'],'local');instrument(ns['cloud'],'cloud_guidance');instrument(cn['client'],'cloud_baseline')
    for method in METHODS:
        plan=plans[method]
        selected=plan['changed']+plan['controls'];random.Random(42).shuffle(selected)
        def one(i,phase):
            item=items[method][i];state.calls=[]
            result=functions[method](item)
            error=result.get('error')
            if error: result['error']='request_failed'  # no credential-bearing exception strings
            if not error and method=='caliper_clf' and result['route']!=item['expected_branch']:
                result['error']='classifier_branch_mismatch'
            if not error and any(not c['response_content'].strip() for c in state.calls):
                result['error']='empty_response'
            return dict(item,**result,method=method,phase=phase,role='replacement' if i in plan['changed'] else 'control',
                calls=state.calls,measured_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        for phase,indices in [('warmup',list(range(20))),('measured',selected)]:
            count=0
            with (dest/f'{method}_{phase}.jsonl').open('x') as f,concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                for r in ex.map(lambda i:one(i,phase),indices):
                    f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();count+=1
                    if count%20==0 or r.get('error'):print('PROGRESS',dataset,method,phase,count,r.get('error'),flush=True)
        print('METHOD_DONE',dataset,method,len(selected),flush=True)
    errors=[]
    for p in dest.glob('*_measured.jsonl'):
        errors += [(p.name,r['index'],r['error']) for r in map(json.loads,p.read_text().splitlines()) if r.get('error')]
    manifest.update(status='COMPLETED_CANDIDATE_NOT_FROZEN' if not errors else 'COMPLETED_WITH_ERRORS',errors=errors)
    dump(dest/'manifest.json',manifest);print('DATASET_DONE',dataset,'errors',errors,flush=True)

if __name__=='__main__':run(sys.argv[1])
