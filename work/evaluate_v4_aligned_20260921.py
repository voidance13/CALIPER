"""CPU-only exact-set V4 audit and paired analysis; never shrink the cohort."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from finalize_router_e2e_20260918 import BASE, EXPECTED, canonical, vals, score, bootstrap, sha

RUN = BASE / 'router_e2e_runs/v4_v3aligned_20260921'

def unique(rows, label):
    keys = [canonical(r['question']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f'Duplicate normalized questions: {label}')
    return dict(zip(keys, rows))

def evaluate(audit_only=False):
    manifest = json.loads((RUN/'manifest.json').read_text())
    result = {'status':'CANDIDATE_NOT_FROZEN', 'datasets':{}, 'evaluator_sha256':sha(Path(__file__)),
              'metric_source_sha256':sha(BASE/'work/finalize_router_e2e_20260918.py'),
              'statistics':'10,000 paired question bootstrap resamples; percentile CI; two tails plus-one correction; seed 42; exploratory unadjusted p-values'}
    for d,n in EXPECTED.items():
        input_path=RUN/f'{d}_inputs.json'
        assert sha(input_path)==manifest['datasets'][d]['input_sha256']
        inputs=json.loads(input_path.read_text())
        cohort=unique(inputs,d)
        assert len(cohort)==n
        refdir=BASE/'router_e2e_runs'/('text_grouped_20260920_unified_bgebase' if d=='musique' else 'text_grouped_20260918')/'predictions'
        references={}
        evidence={str(input_path):sha(input_path)}
        tests=json.loads((BASE/f'data/{d}/test.json').read_text())
        gold={}
        for row in inputs:
            t=tests[row['row_id']]
            assert t['question']==row['question']
            g=t.get('answer') if isinstance(t.get('answer'),str) else (t.get('answers') or [''])[0]
            assert isinstance(g,str) and g.strip()
            gold[canonical(row['question'])]=g
        for v in ('v1','v3'):
            p=refdir/f'{d}_{v}_router_textgrouped_20260918.json'
            references[v]=unique(json.loads(p.read_text()),str(p))
            assert set(references[v])==set(cohort), f'{d}/{v} question mismatch'
            evidence[str(p)]=sha(p)
        if audit_only:
            result['datasets'][d]={'n':n,'reference_alignment':'PASS','evidence':evidence}
            continue
        rows=[]
        for g in range(4):
            p=RUN/f'{d}_shard_{g}.jsonl'
            shard=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
            assert {canonical(r['question']) for r in shard}=={canonical(r['question']) for r in inputs[g::4]}
            rows.extend(shard)
            evidence[str(p)]=sha(p)
        mapped=unique(rows,d+' V4')
        assert set(mapped)==set(cohort)
        ordered=[mapped[canonical(r['question'])] for r in inputs]
        assert all(isinstance(r['pred'],str) for r in ordered)
        metrics,vector=score(ordered,d,gold)
        pairs={}
        refmetrics={}
        for v,mapping in references.items():
            rr=[mapping[canonical(r['question'])] for r in inputs]
            refmetrics[v],rv=score(rr,d,gold)
            pairs[v+'_minus_v4']=bootstrap(rv,vector,42)
        result['datasets'][d]={'metrics':metrics,'reference_metrics':refmetrics,'comparisons':pairs,
                              'empty_predictions':sum(not r['pred'].strip() for r in ordered),
                              'empty_policy':'retained and scored by frozen evaluator; never dropped','evidence':evidence}
        (RUN/f'{d}_v4_predictions.json').write_text(json.dumps(ordered,ensure_ascii=False),encoding='utf-8')
        per=[{'question':r['question'],'gold':gold[canonical(r['question'])],'v4_f1':float(vector[i]),
              **{v+'_f1':float(vals(mapping[canonical(r['question'])]['pred'],gold[canonical(r['question'])])[3]) for v,mapping in references.items()}} for i,r in enumerate(ordered)]
        (RUN/f'{d}_paired_scores.json').write_text(json.dumps(per,ensure_ascii=False),encoding='utf-8')
    target=RUN/('evaluation_preflight.json' if audit_only else 'aligned_v4_metrics_bootstrap.json')
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--wait',action='store_true')
    a=p.parse_args()
    if a.wait:
        deadline=time.time()+6*3600
        while not (RUN/'GENERATION_COMPLETE.json').exists():
            if time.time()>deadline:
                raise TimeoutError('Generation incomplete after six hours; no evaluation published')
            time.sleep(30)
    evaluate(a.audit_only)
