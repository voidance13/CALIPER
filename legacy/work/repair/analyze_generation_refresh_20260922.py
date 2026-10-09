"""Replace only predeclared changed-context records, preserve controls separately."""
import json
from pathlib import Path
import numpy as np
from supplement_table5_generation_20260922 import ROOT,RUN,OLD,METHODS,read,dump,sha,assemble,prepare,selection
from compose_table5_end_to_end_20260921 import stats,paired_bootstrap

def main():
    out=RUN/'analysis';out.mkdir(exist_ok=False)
    summary={'status':'CANDIDATE_NOT_FROZEN','datasets':{},'scope':'Context-matched selective generation refresh; retrieval and generation measured separately. Controls not substituted or treated as extra samples.'}
    comparisons={};controls={};hashes={};all_errors=[]
    for d in ('hotpotqa','musique','2wikimultihopqa'):
        rows=read(OLD/f'{d}_paired_composition.json');count={m:0 for m in METHODS};controls[d]={}
        if d!='2wikimultihopqa':
            old,items,paths=prepare(d)
            for method in METHODS:
                p=RUN/d/f'{method}_measured.jsonl';hashes[str(p)]=sha(p)
                records=list(map(json.loads,p.read_text().splitlines()))
                changed,stable=selection(old,method)
                assert sorted(r['index'] for r in records)==sorted(changed+stable)
                assert len({r['index'] for r in records})==len(records)
                by={r['index']:r for r in records}
                for r in records:
                    assert r['question']==items[method][r['index']]['question']
                    assert r['context_sha256']==items[method][r['index']]['context_sha256']
                    assert r['new_context']==items[method][r['index']]['new_context']
                    prompts=[c['request']['messages'][-1]['content'] for c in r['calls']]
                    assert any(assemble(r['new_context']) in s for s in prompts),'context not sent to model'
                    if r.get('error'):all_errors.append((d,method,r['index'],r['error']))
                for r in rows:
                    if r['method']!=method or r['index'] not in changed:continue
                    new=by[r['index']]
                    assert new['role']=='replacement' and not new.get('error')
                    r['previous_generation_seconds']=r['historical_generation_seconds']
                    r['refreshed_generation_seconds']=new['seconds']
                    r['seconds']=r['retrieval_seconds_added']+new['seconds']
                    r['generation_context_verified']=True
                    r['generation_source']=str(p);r['generation_measured_at']=new['measured_at']
                    count[method]+=1
                valid=[by[i] for i in stable if not by[i].get('error')]
                if valid:
                    prior=np.array([r['old_generation_seconds'] for r in valid]);now=np.array([r['seconds'] for r in valid])
                    controls[d][method]={'n':len(valid),'old_mean':float(prior.mean()),'new_mean':float(now.mean()),'new_over_old_ratio':float(now.mean()/prior.mean()),'note':'Same-context repeat, detects temporal/environment drift; not a controlled causal estimate.'}
                    if len(valid)>=2:controls[d][method]['paired_old_minus_new']=paired_bootstrap(prior,now)
        summary['datasets'][d]={'conditions':{},'replacement_counts':count,'unresolved_valid_context_mismatches':0}
        for r in rows:
            if r['error'] is None and not r.get('generation_context_verified',False) and not r['old_generation_context_exact_match']:
                summary['datasets'][d]['unresolved_valid_context_mismatches']+=1
        assert summary['datasets'][d]['unresolved_valid_context_mismatches']==0
        for m in METHODS:summary['datasets'][d]['conditions'][m]=stats([r['seconds'] for r in rows if r['method']==m and r['error'] is None])
        maps={m:{r['index']:r['seconds'] for r in rows if r['method']==m and r['error'] is None} for m in METHODS}
        comparisons[d]={}
        for m in ('caliper','caliper_clf','baseline'):
            ids=sorted(set(maps[m])&set(maps['cloud_deep']))
            comparisons[d][m]=paired_bootstrap(np.array([maps['cloud_deep'][i] for i in ids]),np.array([maps[m][i] for i in ids]))
        dump(out/f'{d}_paired_composition.json',rows)
    assert not all_errors,all_errors
    dump(out/'summary.json',summary);dump(out/'paired_bootstrap.json',comparisons);dump(out/'same_context_controls.json',controls);dump(out/'input_hashes.json',hashes)
    print(json.dumps({'summary':summary,'controls':controls},indent=2))
if __name__=='__main__':main()
