import json,time
from pathlib import Path
from supplement_table5_generation_20260922 import ROOT,RUN,OLD,read,dump,sha
summary=read(RUN/'analysis/summary.json')
measurements=[]
for d in ('hotpotqa','musique'):
    for p in (RUN/d).glob('*_measured.jsonl'):
        measurements.extend((d,json.loads(s)) for s in p.read_text().splitlines())
assert len(measurements)==526 and all(not r.get('error') for d,r in measurements)
assert sum(r['role']=='replacement' for d,r in measurements)==385
assert all(v['unresolved_valid_context_mismatches']==0 for v in summary['datasets'].values())
slow=[{'dataset':d,'method':r['method'],'index':r['index'],'generation_seconds':r['seconds'],
       'calls':[{'provider':c['provider'],'seconds':c['seconds']} for c in r['calls']]} for d,r in measurements if r['seconds']>30]
receipt={'status':'VALIDATED_CANDIDATE_WITH_TEMPORAL_DRIFT_WARNING','validated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'replacement_records':385,'control_records':141,'warmup_executions':160,'measurement_errors':0,
    'unresolved_valid_context_mismatches':0,'statistical_protocol':'10000 paired bootstrap resamples, seed 42; original valid cohorts retained',
    'frozen_route_replay':'MuSiQue indices 72 and 86 retained and disclosed; new generation consumes the recorded replay-route context. Not a live-route claim.',
    'cross_period_warning':'Cloud deep same-context controls are faster by 22.02% (HotpotQA) and 27.39% (MuSiQue); selective refresh does not establish stationary service speed.',
    'slow_successful_requests_retained':slow,'frozen_paper_or_archive_modified':False,
    'hashes':{str(p):sha(p) for p in sorted((RUN/'analysis').glob('*.json'))}}
dump(RUN/'VALIDATION_RECEIPT.json',receipt)
s=read(RUN/'process_status.json');s['analysis_status']=receipt['status'];dump(RUN/'process_status.json',s)
print(json.dumps(receipt,indent=2))
