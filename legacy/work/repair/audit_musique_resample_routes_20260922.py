import json
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path('/mnt/data/caliper/code/CALIPER')
selection=json.loads((ROOT/'latency_runs/musique_resample_seed20260922_v1/SEALED_SAMPLE.json').read_text())
snapshot=ROOT/'router_e2e_runs/text_grouped_20260920_unified_bgebase/routed_records.jsonl'
byq={}
counts=Counter()
for line in snapshot.open():
    r=json.loads(line)
    if r['dataset'] in ('musique','ms'):
        byq[r['question']]=r
        counts[(r['predicted_route'],r['model']['embedder'],r['model']['reranker'])]+=1
import sys;sys.path.insert(0,str(ROOT))
import fasttext,jieba
source=fasttext.load_model(str(ROOT/'query_clf/text_grouped_source_router_20260917/domain_clf.bin'))
stop=set((ROOT/'query_clf/stopwords.txt').read_text().splitlines())
def route(q):
    text=' '.join(s for s in jieba.lcut(q) if len(s)>1 and s not in stop)
    return source.predict(' '.join(text.lower().split()))[0][0].removeprefix('__label__')
mismatches=[];livecounts=Counter()
for row in selection['selected']:
    q=row['question'];live=route(q);frozen=byq[q]['predicted_route'];livecounts[live]+=1
    if live!=frozen:mismatches.append({'draw_position':row['draw_position'],'source_index':row['source_index'],'live':live,'snapshot':frozen})
print(json.dumps({'selected_route_counts':livecounts,'mismatch_n':len(mismatches),'mismatches':mismatches,'snapshot_model_counts':[{"route":k[0],"embedder":k[1],"reranker":k[2],"n":v} for k,v in counts.items()]},indent=2,default=dict))
