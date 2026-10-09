"""Branch-matched retrieval + historical generation estimates, never integrated timings."""
import json
import hashlib
from pathlib import Path
import numpy as np
from compose_table5_end_to_end_20260921 import stats, paired_bootstrap, seconds

ROOT=Path('/mnt/data/caliper/code/CALIPER')
OUT=ROOT/'latency_runs/composed_table5_20260922_v1'
DATASETS=('hotpotqa','musique','2wikimultihopqa')
METHODS=('baseline','caliper_clf','caliper','cloud_deep')

def overhead(method,branch,fixed,core,clf):
    if method=='caliper': return core['total_seconds']
    if method=='caliper_clf':
        if branch != clf['classifier_branch']:raise ValueError('classifier branch mismatch')
        if branch=='local':return fixed['total_seconds']
        if branch!='collaborative':raise ValueError('unknown classifier branch')
        return clf['total_seconds']-clf['classifier_seconds']
    return fixed['total_seconds']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    summary={'status':'CANDIDATE_COMPOSED_NOT_INTEGRATED','datasets':{}}
    bootstrap={'bootstrap_samples':10000,'seed':42,'datasets':{}}
    files={}
    for d in DATASETS:
        version='v2' if d=='musique' else 'v1'
        source=ROOT/f'retrieval_runs/path_latency_per_question_20260922_{version}/{d}'
        retrieval={}
        for name in ('fixed_reference','routed_core','routed_clf'):
            p=source/f'{name}.jsonl'; files[str(p)]=sha(p)
            retrieval[name]=[json.loads(s) for s in p.read_text().splitlines()]
        raw=[]
        for name in (f'table4_timing_{d}_seed42.json',f'table4_clf_actual_{d}_seed42.json',f'table4_cloud_deep_{d}_full_b256_seed42.json'):
            p=ROOT/'table-result'/name;files[str(p)]=sha(p);raw.append(load(p)['rows'])
        assert all(len(rows)==200 for rows in [*retrieval.values(),*raw])
        records=[]
        for i in range(200):
            timing,actual,cloud=[rows[i] for rows in raw]
            fixed,core,clf=[retrieval[k][i] for k in ('fixed_reference','routed_core','routed_clf')]
            q=timing['question']
            assert all(r['question']==q for r in (actual,cloud,fixed,core,clf))
            assert all(r['index']==i and r['error'] is None for r in (fixed,core,clf))
            values={'baseline':seconds(timing['baseline']),'caliper':seconds(timing['caliper']),
                    'caliper_clf':seconds(actual),'cloud_deep':seconds(cloud)}
            for method,generation in values.items():
                selected=core if method=='caliper' else clf if method=='caliper_clf' and actual['route']=='collaborative' else fixed
                extra=overhead(method,actual['route'],fixed,core,clf)
                assert extra>0
                records.append({'dataset':d,'index':i,'question':q,'method':method,
                    'classifier_branch':actual['route'] if method=='caliper_clf' else None,
                    'retrieval_seconds_added':extra,'historical_generation_seconds':generation,
                    'seconds':None if generation is None else generation+extra,
                    'error':None if generation is not None else 'historical_generation_error',
                    'old_generation_context_exact_match':selected['matches_old_generation_context'],
                    'frozen_route_replayed':selected.get('frozen_route_replayed',False)})
        (OUT/f'{d}_paired_composition.json').write_text(json.dumps(records,indent=2))
        summary['datasets'][d]={'conditions':{},'context_matches':{},'route_replays':{}}
        for method in METHODS:
            rows=[r for r in records if r['method']==method]
            summary['datasets'][d]['conditions'][method]=stats([r['seconds'] for r in rows if r['error'] is None])
            summary['datasets'][d]['context_matches'][method]=sum(r['old_generation_context_exact_match'] for r in rows)
            summary['datasets'][d]['route_replays'][method]=sum(r['frozen_route_replayed'] for r in rows)
        common=[i for i in range(200) if all(r['error'] is None for r in records if r['index']==i and r['method'] in ('caliper','cloud_deep'))]
        def values(method):return np.array([r['seconds'] for r in records if r['method']==method and r['index'] in common])
        bootstrap['datasets'][d]=paired_bootstrap(values('cloud_deep'),values('caliper'))
    for n,obj in [('summary.json',summary),('paired_bootstrap.json',bootstrap),('input_hashes.json',files)]:
        (OUT/n).write_text(json.dumps(obj,indent=2))
    print(json.dumps({'summary':summary,'bootstrap':bootstrap},indent=2))

if __name__=='__main__':main()
