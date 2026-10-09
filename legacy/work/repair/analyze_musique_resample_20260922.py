"""Validate sealed MuSiQue resample and compose fresh retrieval/generation timings."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from supplement_table5_generation_20260922 import ROOT,METHODS,assemble
from compose_table5_end_to_end_20260921 import stats,paired_bootstrap
RUN=ROOT/'latency_runs/musique_resample_seed20260922_v1';RETR=RUN/'retrieval_live_current/musique';GEN=RUN/'generation';OUT=RUN/'analysis'
def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def jsonl(path):return list(map(json.loads,path.read_text().splitlines()))
def main():
    selection=read(RUN/'SEALED_SAMPLE.json');manifest=read(GEN/'manifest.json')
    assert selection['status']=='SEALED_BEFORE_MEASUREMENT' and manifest['status'] in ('MEASUREMENTS_COMPLETED_PENDING_VALIDATION','COMPLETED_WITH_ERRORS')
    for name,value in manifest['input_sha256'].items():assert sha(Path(name))==value,'input changed'
    OUT.mkdir(exist_ok=False)
    retrieval={name:{r['index']:r for r in jsonl(RETR/f'{name}.jsonl')} for name in ('fixed_reference','routed_core','routed_clf')}
    assert all(set(rows)==set(range(200)) for rows in retrieval.values())
    composed=[];failures=[];slow=[];models=set();attempts=0;warmups=0
    for method in METHODS:
        warm=jsonl(GEN/f'{method}_warmup.jsonl');assert len(warm)==20;warmups+=20
        assert all(not r.get('error') for r in warm),'warmup error'
        rows=jsonl(GEN/f'{method}_measured.jsonl');assert len(rows)==200 and sorted(r['index'] for r in rows)==list(range(200))
        for row in rows:
            attempts+=1;i=row['index'];selected=selection['selected'][i]
            assert row['question']==selected['question'] and row['source_index']==selected['source_index']
            if row.get('error'):
                failures.append({'method':method,'index':i,'source_index':row['source_index'],'error':row['error']});continue
            expected=['cloud_baseline']*2 if method=='cloud_deep' else ['local'] if method=='baseline' or (method=='caliper_clf' and row['expected_branch']=='local') else ['cloud_guidance','local','local']
            assert [c['provider'] for c in row['calls']]==expected
            assert any(assemble(row['new_context']) in c['request']['messages'][-1]['content'] for c in row['calls'])
            for call in row['calls']:
                models.add(str(call['model']));request=call['request']
                assert request['temperature']==(.1 if call['provider']=='local' else .2)
                assert 'seed' not in request
                assert request['model']==('/mnt/data/caliper/models/Qwen3.5-9B' if call['provider']=='local' else 'qwen3.7-plus')
            tokens=[c['request']['max_tokens'] for c in row['calls']]
            assert tokens==([256,64] if method=='cloud_deep' else [128] if expected==['local'] else [64,96,32])
            condition=row['retrieval_condition'];rr=retrieval[condition][i]
            assert rr['question']==row['question'] and rr['context']==row['new_context']
            overhead=rr['total_seconds']-(rr['classifier_seconds'] if method=='caliper_clf' and condition=='routed_clf' else 0.)
            total=overhead+row['seconds'];assert overhead>0 and row['seconds']>0 and np.isfinite(total)
            record={'index':i,'source_index':row['source_index'],'question':row['question'],'method':method,
                'retrieval_condition':condition,'retrieval_seconds':overhead,'generation_seconds':row['seconds'],'seconds':total,
                'context_sha256':row['context_sha256'],'measured_at':row['measured_at']}
            composed.append(record)
            if row['seconds']>30:slow.append({'method':method,'index':i,'source_index':row['source_index'],'generation_seconds':row['seconds'],'calls':[{'provider':c['provider'],'seconds':c['seconds']} for c in row['calls']]})
    assert attempts==800 and warmups==80
    conditions={m:stats([r['seconds'] for r in composed if r['method']==m]) for m in METHODS}
    generation={m:stats([r['generation_seconds'] for r in composed if r['method']==m]) for m in METHODS}
    maps={m:{r['index']:r['seconds'] for r in composed if r['method']==m} for m in METHODS}
    paired={}
    for method in ('baseline','caliper','caliper_clf'):
        ids=sorted(set(maps[method])&set(maps['cloud_deep']))
        paired[method]=paired_bootstrap(np.array([maps['cloud_deep'][i] for i in ids]),np.array([maps[method][i] for i in ids]))
    summary={'status':'VALIDATED_CANDIDATE_NOT_FROZEN','dataset':'musique','sample_seed':selection['seed'],'conditions':conditions,'generation_only':generation,
        'attempted':attempts,'successful':len(composed),'failures':failures,'slow_successes':slow,'paired_bootstrap_vs_cloud':paired,
        'scope':'Fresh random sample and fresh generation; retrieval and generation separately measured and composed per question, not integrated live execution.'}
    dump(OUT/'paired_composition.json',composed);dump(OUT/'summary.json',summary)
    receipt={'status':'VALIDATED_CANDIDATE_NOT_FROZEN','validated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'sample_sealed_before_measurement':True,'sample_seed':selection['seed'],'attempted':attempts,'successful':len(composed),'failures':failures,
        'warmups':warmups,'context_and_request_checks':'PASS','historical_generation_seconds_used':False,'outcome_based_replacement':False,
        'response_models':sorted(models),'slow_successes_retained':slow,'paper_archive_modified':False,
        'limitations':['Retrieval and generation are composed per question, not integrated live execution.','Generation request seed remains omitted to match the current Table 5 protocol.','External provider latency can vary by measurement period.'],
        'output_sha256':{'summary.json':sha(OUT/'summary.json'),'paired_composition.json':sha(OUT/'paired_composition.json')}}
    dump(RUN/'VALIDATION_RECEIPT.json',receipt);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
