"""Verify and compose one all-new generation round; never use old generation durations."""
import json,time
from pathlib import Path
import numpy as np
from full_generation_table5_20260922 import ROOT,RUN,DATASETS,METHODS,prepare,sha,dump,assemble
from supplement_table5_generation_20260922 import OLD,read
from compose_table5_end_to_end_20260921 import stats,paired_bootstrap

def analyze():
    manifest=read(RUN/'manifest.json')
    assert manifest['status'] in ('MEASUREMENTS_COMPLETED_PENDING_VALIDATION','COMPLETED_WITH_ERRORS')
    for d,m,i,phase,error in manifest['errors']:
        assert phase=='measured' and i not in manifest['primary_valid_indices'][d][m], 'Unexpected primary or warmup failure'
    for name,value in manifest['input_sha256'].items():assert sha(Path(name))==value,'input changed'
    out=RUN/'analysis';out.mkdir(exist_ok=False)
    summary={'status':'VALIDATED_CANDIDATE_NOT_FROZEN','datasets':{},'scope':manifest['scope']}
    all200={};paired={};slow=[];hashes={};total=0;warm=0;response_models=set();failures=[];attempts=0
    for d in DATASETS:
        old,items,paths=prepare(d)
        overhead={(r['method'],r['index']):r for r in old}
        composed=[];all200[d]={}
        for m in METHODS:
            p=RUN/'measurements'/f'{d}_{m}_measured.jsonl'
            rows=list(map(json.loads,p.read_text().splitlines()));hashes[str(p)]=sha(p)
            assert len(rows)==200 and sorted(r['index'] for r in rows)==list(range(200))
            wp=RUN/'measurements'/f'{d}_{m}_warmup.jsonl';wr=list(map(json.loads,wp.read_text().splitlines()))
            assert len(wr)==20 and all(not r.get('error') for r in wr);warm+=len(wr)
            for r in rows:
                i=r['index'];expect=items[m][i]
                attempts+=1
                assert r['dataset']==d and r['method']==m
                assert r['question']==expect['question'] and r['new_context']==expect['new_context'] and r['context_sha256']==expect['context_sha256']
                assert r['primary_included']==(i in manifest['primary_valid_indices'][d][m])
                if r.get('error'):
                    assert not r['primary_included'], 'Primary-cohort failure'
                    failures.append({'dataset':d,'method':m,'index':i,'error':r['error'],'excluded_before_run':True,'source':str(p)})
                    continue
                assert any(assemble(r['new_context']) in c['request']['messages'][-1]['content'] for c in r['calls'])
                providers=[c['provider'] for c in r['calls']]
                expected=['cloud_baseline']*2 if m=='cloud_deep' else ['local'] if m=='baseline' or (m=='caliper_clf' and r['expected_branch']=='local') else ['cloud_guidance','local','local']
                assert providers==expected,(d,m,i,providers,expected)
                for c in r['calls']:
                    response_models.add(str(c['model']))
                    request=c['request'];assert request['temperature']==(.1 if c['provider']=='local' else .2)
                    assert 'seed' not in request
                    assert request['model']==('/mnt/data/caliper/models/Qwen3.5-9B' if c['provider']=='local' else 'qwen3.7-plus')
                    if c['provider']=='local':assert request['extra_body']['chat_template_kwargs']['enable_thinking'] is False
                    elif c['provider']=='cloud_guidance':assert request['extra_body']['enable_thinking'] is False
                tokens=[c['request']['max_tokens'] for c in r['calls']]
                assert tokens==([256,64] if m=='cloud_deep' else [128] if providers==['local'] else [64,96,32])
                if m=='cloud_deep':
                    assert r['calls'][0]['request']['extra_body']=={'enable_thinking':True,'thinking_budget':256}
                    assert r['calls'][1]['request']['extra_body']=={'enable_thinking':False}
                extra=overhead[m,i]['retrieval_seconds_added'];assert extra>0 and r['seconds']>0
                assert np.isfinite(extra+r['seconds'])
                composed.append({'dataset':d,'index':i,'question':r['question'],'method':m,
                    'primary_included':r['primary_included'],'retrieval_seconds':extra,'generation_seconds':r['seconds'],'seconds':extra+r['seconds'],
                    'context_sha256':r['context_sha256'],'generation_context_verified':True,'frozen_route_replayed':r['frozen_route_replayed'],
                    'generation_source':str(p),'retrieval_source':r['retrieval_record'],'measured_at':r['measured_at']})
                if r['seconds']>30:slow.append({'dataset':d,'method':m,'index':i,'generation_seconds':r['seconds'],'calls':[{'provider':c['provider'],'seconds':c['seconds']} for c in r['calls']]})
                total+=1
            cell=[r['seconds'] for r in composed if r['method']==m]
            all200[d][m]=({'status':'AVAILABLE','statistics':stats(cell)} if len(cell)==200 else
                {'status':'UNAVAILABLE_INCOMPLETE','attempted':200,'successful':len(cell),'imputed':False})
        valid=[r for r in composed if r['primary_included']]
        for m in METHODS:
            assert sorted(r['index'] for r in valid if r['method']==m)==sorted(manifest['primary_valid_indices'][d][m])
        summary['datasets'][d]={'conditions':{m:stats([r['seconds'] for r in valid if r['method']==m]) for m in METHODS},
            'generation_only':{m:stats([r['generation_seconds'] for r in valid if r['method']==m]) for m in METHODS}}
        maps={m:{r['index']:r['seconds'] for r in valid if r['method']==m} for m in METHODS}
        paired[d]={}
        for m in ('caliper','caliper_clf','baseline'):
            ids=sorted(set(maps[m])&set(maps['cloud_deep']))
            paired[d][m]=paired_bootstrap(np.array([maps['cloud_deep'][i] for i in ids]),np.array([maps[m][i] for i in ids]))
        dump(out/f'{d}_paired_composition.json',composed)
    assert attempts==2400 and total+len(failures)==attempts and warm==240
    assert {(r['dataset'],r['method'],r['index']) for r in failures}=={(d,m,i) for d,m,i,phase,error in manifest['errors']}
    dump(out/'summary.json',summary);dump(out/'all_200_sensitivity.json',all200);dump(out/'paired_bootstrap.json',paired)
    dump(out/'input_hashes.json',hashes)
    receipt={'status':'VALIDATED_CANDIDATE_NOT_FROZEN','validated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'attempted_measured_records':attempts,'successful_measured_records':total,'warmup_records':warm,
      'primary_records':sum(v['conditions'][m]['n'] for v in summary['datasets'].values() for m in METHODS),
      'errors':len(failures),'expected_preexcluded_failures':failures,'unexpected_primary_failures':0,'all_generation_records_new':True,'context_and_request_parameter_checks':'PASS','historical_generation_seconds_used':False,
      'analysis_script_sha256':sha(Path(__file__)),'response_model_names':sorted(response_models),
      'slow_successful_requests_retained':slow,'limitations':['Retrieval durations retained from a separate measurement period; composition is an estimate, not integrated live measurement.',
      'Table 5 generation configuration differs from Table 2 quality configuration.','Two disclosed MuSiQue frozen-route replays remain.',
      'Interleaving reduces time-of-day confounding but does not eliminate provider variability.'],
      'paper_and_archive_modified':False,'analysis_sha256':{str(p):sha(p) for p in sorted(out.glob('*.json'))}}
    dump(RUN/'VALIDATION_RECEIPT.json',receipt)
    print(json.dumps({'summary':summary,'paired_bootstrap':paired,'receipt':receipt},indent=2))
if __name__=='__main__':analyze()
