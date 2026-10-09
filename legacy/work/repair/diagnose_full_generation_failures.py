import json,re
from full_generation_table5_20260922 import ROOT,RUN
from supplement_table5_generation_20260922 import read
m=read(RUN/'manifest.json');p=read(RUN/'progress.json')
for d,method,i,phase,error in p['errors']:
    name=f'table4_timing_{d}_seed42.json' if method in ('baseline','caliper') else f'table4_clf_actual_{d}_seed42.json' if method=='caliper_clf' else f'table4_cloud_deep_{d}_full_b256_seed42.json'
    row=read(ROOT/'table-result'/name)['rows'][i]
    value=row.get(method) if method in ('baseline','caliper') else row
    old_error=value.get('error') if isinstance(value,dict) else None
    # Emit categorical evidence only, never credential-bearing exception bodies.
    text=str(old_error or '')
    flags={s:s.lower() in text.lower() for s in ('400','401','429','DataInspection','inappropriate','sensitive','timeout','content_filter','risk')}
    print(json.dumps({'dataset':d,'method':method,'index':i,'phase':phase,'error':error,
        'excluded_before_run':i not in m['primary_valid_indices'][d][method],
        'historical_error_recorded':bool(old_error),'historical_error_flags':flags}))
