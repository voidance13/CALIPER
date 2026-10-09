"""Recompute all routed branches with unified backbones; no old scores reused."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from collections import Counter

ROOT = Path('/mnt/data/caliper/code/CALIPER')
RUN = ROOT / 'retrieval_runs/routed_large_20260921'
HELPER = ROOT / 'work/repair/true_routed_recall_heldout.py'
ROUTER = ROOT / 'query_clf/text_grouped_source_router_20260917/domain_clf.bin'
OLD = Path('/tmp/true_routed_unified_bgebase_20260920/final_records/all_records.jsonl')
ns = {'__name__': 'routed_helpers'}
exec(compile(HELPER.read_text().split('parser = argparse.ArgumentParser()')[0], str(HELPER), 'exec'), ns)
MODELS = {
    'hp': ('hotpotqa_bge-embedder-ft', 'unified_large_ftsep_hotpotqa'),
    'ms': ('unified_bgebase_musique_firstpos_embedder', 'unified_large_ftsep_musique'),
    'tw': ('2wikimultihopqa_bge-embedder-ft', 'unified_large_ftsep_2wiki'),
}

def worker(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain.retrievers import ContextualCompressionRetriever
    from BCERerank import BCERerank
    import nltk
    import torch
    import gc
    plan = json.loads((RUN/'plan.json').read_text())
    _, predict = ns['make_router'](ROUTER, normalize_input=True)
    data = {d: {r['row_id']: r for r in ns['load_rows'](d)} for d in ('hp','ms','tw')}
    selected = plan[gpu::4]
    done = 0
    started = time.time()
    with (RUN/f'shard_{gpu}.jsonl').open('x') as out:
        for route, (ename, rname) in MODELS.items():
            paths = {k: str(ROOT/'retrieve_fine_tuning/models'/v) for k,v in [('embedder',ename),('reranker',rname)]}
            emb = HuggingFaceBgeEmbeddings(model_name=paths['embedder'], model_kwargs={'device':'cuda:0'},
                encode_kwargs={'normalize_embeddings':True}, query_instruction='Represent this sentence for searching relevant passages:')
            rer = BCERerank(model=paths['reranker'],top_n=10,device='cuda:0',use_fp16=True)
            for task in selected:
                if task['predicted_route'] != route:
                    continue
                d, idx = task['dataset'], task['row_id']
                row = data[d][idx]
                assert row['question'] == task['question'] and predict(row['question']) == route
                # Preserve the already adopted ms->ms tokenizer; other branches
                # retain the historical helper's context construction unchanged.
                if d == 'ms' and route == 'ms':
                    texts = [s for p in row['raw']['paragraphs'] for s in nltk.sent_tokenize(p['paragraph_text']) if s.strip()]
                else:
                    texts = [s for c in row['ctxs'] for s in c['sentences'] if s.strip()]
                index = FAISS.from_texts(texts=texts,embedding=emb,distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
                comp = ContextualCompressionRetriever(base_compressor=rer,
                    base_retriever=index.as_retriever(search_type='similarity',search_kwargs={'k':20}))
                retrieved = [doc.page_content for doc in comp.get_relevant_documents(row['question'])]
                scores = ns['recall_for'](d,row,retrieved)
                assert retrieved and scores is not None
                result = dict(task,model=paths,retrieved_text=retrieved,recall=scores,error=None)
                out.write(json.dumps(result,ensure_ascii=False)+'\n')
                out.flush()
                done += 1
                if done % 100 == 0:
                    print(gpu,done,len(selected),round(time.time()-started),flush=True)
            del comp,index,emb,rer
            gc.collect()
            torch.cuda.empty_cache()
    assert done == len(selected)
    print('SHARD_DONE',gpu,done,flush=True)

def main():
    RUN.mkdir(parents=True,exist_ok=False)
    old = [json.loads(s) for s in OLD.read_text().splitlines() if s.strip()]
    plan = [{k:r[k] for k in ('dataset','row_id','question','predicted_route')} for r in old]
    plan.sort(key=lambda r:(r['dataset'],r['row_id']))
    assert Counter(r['dataset'] for r in plan)=={'hp':7405,'ms':4293,'tw':12576}
    assert len({(r['dataset'],r['row_id']) for r in plan})==24274
    for d in ('hp','ms','tw'):
        assert {r['row_id'] for r in plan if r['dataset']==d} == {r['row_id'] for r in ns['load_rows'](d)}
    (RUN/'plan.json').write_text(json.dumps(plan))
    sha=ns['sha256']
    evidence=[OLD,HELPER,ROUTER,ns['STOPWORDS'],ns['MU_DEV'],Path(__file__)]
    evidence.extend(ROOT/f'data/{d}/test.json' for d in ('hotpotqa','2wikimultihopqa'))
    for pair in MODELS.values():
        for model in pair:
            directory=ROOT/'retrieve_fine_tuning/models'/model
            weights=list(directory.glob('*.safetensors'))+list(directory.glob('pytorch_model*.bin'))
            assert weights
            evidence.extend(weights+[directory/'config.json'])
    manifest={'status':'RUNNING_NOT_FROZEN','started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'model_mapping':MODELS,'sha256':{str(p):sha(p) for p in evidence},
        'scope':'All 24274 questions and all routes recomputed; same router decisions checked live; no cloud calls',
        'protocol_note':'Existing route-specific sentence preprocessing retained, including NLTK on ms->ms; no tokenization redesign.'}
    (RUN/'manifest.json').write_text(json.dumps(manifest,indent=2))
    processes=[]
    handles=[]
    for gpu in range(4):
        handle=(RUN/f'shard_{gpu}.log').open('x'); handles.append(handle)
        processes.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--gpu',str(gpu)],
            stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')))
    deadline=time.time()+8*3600
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None,0) for p in processes):
            raise RuntimeError('Worker failed; no automatic retry; inspect shard logs')
        if time.time()>deadline:
            for p in processes:
                if p.poll() is None: p.terminate()
            raise TimeoutError('Eight-hour limit')
        time.sleep(10)
    assert all(p.returncode==0 for p in processes)
    for h in handles: h.close()
    records=[json.loads(s) for gpu in range(4) for s in (RUN/f'shard_{gpu}.jsonl').read_text().splitlines()]
    keys=[(r['dataset'],r['row_id']) for r in records]
    assert len(keys)==len(set(keys))==24274
    assert set(keys)=={(r['dataset'],r['row_id']) for r in plan}
    result={'status':'COMPLETED_CANDIDATE_NOT_FROZEN','completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'n':len(records),'errors':0,'datasets':{},'shard_sha256':{str(p):sha(p) for p in RUN.glob('shard_*.jsonl')}}
    for d in ('hp','ms','tw'):
        rows=[r for r in records if r['dataset']==d]
        prev=[r for r in old if r['dataset']==d]
        result['datasets'][d]={'n':len(rows),'routes':dict(Counter(r['predicted_route'] for r in rows)),
            'new':{f'R@{k}':round(statistics.mean(r['recall'][f'R@{k}'] for r in rows)*100,2) for k in (3,5,10)},
            'old':{f'R@{k}':round(statistics.mean(r['recall'][f'R@{k}'] for r in prev)*100,2) for k in (3,5,10)}}
    (RUN/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--gpu',type=int,choices=range(4))
    args=parser.parse_args()
    main() if args.gpu is None else worker(args.gpu)
