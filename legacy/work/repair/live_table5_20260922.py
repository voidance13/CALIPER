"""Integrated, uncached Table 5 candidate. Never writes frozen evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import queue
import time

ROOT = Path('/mnt/data/caliper/code/CALIPER')
MODEL = '/mnt/data/caliper/models/Qwen3.5-9B'
METHODS = ('baseline', 'caliper', 'caliper_clf', 'cloud')


def single_gpu_reranker(factory, path, device):
    model = factory(path, use_fp16=True, devices=[device])
    if model.target_devices != [device]:
        raise RuntimeError(f'Reranker binding mismatch: {model.target_devices} != {[device]}')
    return model


def measure_one(question, method, retrieve, generate, clock=time.perf_counter):
    start = clock()
    retrieved = retrieve(question, method)
    retrieval_end = clock()
    generated = generate(question, retrieved['context'], method, retrieved.get('branch'))
    end = clock()
    return dict(retrieved, **generated, seconds=end-start,
                retrieval_seconds=retrieval_end-start, generation_seconds=end-retrieval_end)


def main(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import sys
    sys.path.insert(0, str(ROOT))
    import fasttext
    import jieba
    import nltk
    import numpy as np
    import torch
    from openai import OpenAI
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain.retrievers import ContextualCompressionRetriever
    from BCERerank import BCERerank
    from sentence_transformers import SentenceTransformer
    from FlagEmbedding import FlagReranker
    from measure_route_latency_20260922 import align_rows, contexts, digest, read
    torch.set_num_threads(2)
    nltk.data.path.append('/home/gpt/nltk_data')
    jieba.initialize()
    dest = args.output / args.dataset
    dest.mkdir(parents=True, exist_ok=False)
    for line in (ROOT / '.env').read_text().splitlines():
        if line.strip() and '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    cloud = OpenAI(api_key=os.environ['ALIYUN_API_KEY'],
                   base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', timeout=180, max_retries=0)
    local = OpenAI(api_key='EMPTY', base_url=f'http://127.0.0.1:{args.port}/v1', timeout=180, max_retries=0)
    models_dir = ROOT / 'retrieve_fine_tuning/models'
    # Main E2E specialist families, not Table 8 full-train models.
    names = {
        'hp': ('hotpotqa_bge-embedder-ft', 'hotpotqa_bge-reranker-v2-m3-ft'),
        'ms': ('unified_bgebase_musique_firstpos_embedder', 'unified_large_ftsep_musique'),
        'tw': ('2wikimultihopqa_bge-embedder-ft', '2wikimultihopqa_bge-reranker-v2-m3-ft_v2'),
    }
    paths = {k: tuple(str(models_dir / p) for p in v) for k, v in names.items()}
    aliases = {'hotpotqa': 'hp', 'musique': 'ms', '2wikimultihopqa': 'tw'}
    timing_path = ROOT / f'table-result/table4_timing_{args.dataset}_seed42.json'
    tests_path = ROOT / f'data/{args.dataset}/test.json'
    old_rows = read(timing_path)['rows']
    assert len(old_rows) == 200
    sample = align_rows([r['question'] for r in old_rows], read(tests_path))[:args.limit]
    assert len({r['question'] for r in sample}) == len(sample), 'ambiguous duplicate sample questions'
    by_question = {r['question']: r for r in sample}
    source_path = ROOT / 'query_clf/text_grouped_source_router_20260917/domain_clf.bin'
    classifier_path = ROOT / 'query_clf/train_new/9b_clf.bin'
    source = fasttext.load_model(str(source_path))
    classifier = fasttext.load_model(str(classifier_path))
    stop_path = ROOT / 'query_clf/stopwords.txt'
    stopwords = set(stop_path.read_text().splitlines())
    base_paths = ('/mnt/data/caliper/models/bge-base-en-v1.5', '/mnt/data/caliper/models/bge-reranker-large')
    evidence = [Path(__file__), Path(__file__).with_name('measure_route_latency_20260922.py'),
                timing_path, tests_path, source_path, classifier_path, stop_path]
    for directory in set(p for pair in paths.values() for p in pair) | set(base_paths) | {MODEL}:
        p = Path(directory)
        weights = list(p.glob('*.safetensors')) + list(p.glob('pytorch_model*.bin'))
        assert weights, f'missing model weights: {p}'
        evidence.extend(weights + [p / 'config.json'])
    manifest = {
        'status': 'RUNNING_CANDIDATE_NOT_FROZEN', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'dataset': args.dataset, 'n': len(sample), 'warmup': args.warmup, 'workers': 4,
        'source_router': str(source_path), 'classifier': str(classifier_path), 'threshold': .997,
        'specialists': paths, 'base': base_paths, 'local_model': MODEL, 'cloud_model': 'qwen3.7-plus',
        'local_generation': {'temperature': 0, 'seed': 42, 'thinking': False, 'max_tokens': [256, 64]},
        'cloud_guidance': {'temperature': .2, 'thinking': False, 'max_tokens': 64},
        'cloud_baseline': {'temperature': .2, 'thinking_budget': 256, 'max_tokens': [256, 64]},
        'scope': 'live request retrieval through final answer; per-question candidate corpus, not full Wikipedia',
        'hardware': 'GPU0 local generation; retrieval workers on GPU1,2,3,1; datasets sequential',
        'exclusions': ['model loading', 'disk input loading', 'external load-generator waiting before worker starts'],
        'differences_from_frozen_quality': ['live router without overrides', 'fresh cloud instructions and answers',
            'one current first-position MuSiQue specialist for all ms routes; no historical per-question checkpoint selection'],
        'sha256': {str(p): digest(p) for p in dict.fromkeys(evidence)},
    }
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (dest / 'sample.json').write_text(json.dumps([{'index': i, 'question': r['question']} for i,r in enumerate(sample)], indent=2))
    print('MANIFEST_READY', args.dataset, flush=True)
    pool = queue.Queue()
    for worker, gpu in enumerate((1, 2, 3, 1)):
        device = f'cuda:{gpu}'
        specialists = {}
        for key, pair in paths.items():
            emb = HuggingFaceBgeEmbeddings(model_name=pair[0], model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}, query_instruction='Represent this sentence for searching relevant passages:')
            rer = BCERerank(model=pair[1], top_n=10, device=device, use_fp16=True)
            specialists[key] = (emb, rer)
        base = (SentenceTransformer(base_paths[0], device=device).half(),
                single_gpu_reranker(FlagReranker, base_paths[1], device)) if args.dataset == 'musique' else None
        if base is not None:
            print('BASE_RERANKER_DEVICE_VERIFIED', worker, base[1].target_devices, flush=True)
        pool.put((specialists, base))
        print('RETRIEVAL_WORKER_READY', worker, gpu, flush=True)

    def tokenize(q):
        return ' '.join(s for s in jieba.lcut(q) if len(s)>1 and s not in stopwords)

    def retrieve(q, method):
        started = time.perf_counter()
        branch, score = None, None
        if method == 'caliper_clf':
            labels, probs = classifier.predict(tokenize(q), k=2)
            score = next((float(p) for label,p in zip(labels,probs) if label == '__label__easy'), 0.)
            branch = 'local' if score >= .997 else 'collaborative'
        routed = method in ('caliper', 'caliper_clf')
        route = source.predict(' '.join(tokenize(q).lower().split()))[0][0].removeprefix('__label__') if routed else aliases[args.dataset]
        routing_seconds = time.perf_counter()-started
        specialists, base_model = pool.get()
        try:
            texts = [s for c in by_question[q]['ctxs'] for s in c['sentences'] if s.strip()]
            emb, rer = specialists[route]
            index = FAISS.from_texts(texts, emb, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
            retriever = ContextualCompressionRetriever(base_compressor=rer,
                base_retriever=index.as_retriever(search_type='similarity', search_kwargs={'k':20}))
            ft = [d.page_content for d in retriever.get_relevant_documents(q)]
            base = []
            if routed and args.dataset == 'musique':
                import faiss
                emb, rer = base_model
                query = emb.encode([q], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
                vectors = emb.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
                ix = faiss.IndexFlatIP(vectors.shape[1]); ix.add(vectors)
                _, ids = ix.search(query, min(20,len(texts)))
                cand = [texts[i] for i in ids[0]]
                scores = np.atleast_1d(rer.compute_score([[q,s] for s in cand]))
                base = [cand[i] for i in sorted(range(len(cand)), key=lambda i:-float(scores[i]))[:3]]
            return {'context': contexts(args.dataset,ft,base) if routed else ft[:5],
                    'source_route': route, 'branch': branch, 'easy_score': score,
                    'routing_classifier_seconds': routing_seconds, 'retrieval_models': paths[route]}
        finally:
            pool.put((specialists,base_model))

    def generate(q, passages, method, branch):
        ctx = '\n'.join(f'[{i+1}] {s}' for i,s in enumerate(passages))
        calls = []
        def call(client, system, prompt, tokens, thinking=False):
            remote = client is cloud
            kwargs = dict(model='qwen3.7-plus' if remote else MODEL,
                temperature=.2 if remote else 0, max_tokens=tokens,
                messages=[{'role':'system','content':system},{'role':'user','content':prompt}],
                extra_body=({'enable_thinking':thinking, **({'thinking_budget':256} if thinking else {})}
                            if remote else {'chat_template_kwargs':{'enable_thinking':False}}))
            if not remote: kwargs['seed'] = 42
            start = time.perf_counter()
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ''
            if not content.strip(): raise RuntimeError('empty model content')
            calls.append({'provider':'cloud' if remote else 'local','seconds':time.perf_counter()-start,
                'usage':response.usage.model_dump() if response.usage else {}, 'finish_reason':response.choices[0].finish_reason})
            return content
        guidance = None
        if method == 'cloud':
            reasoning = call(cloud, 'You are a helpful assistant. According to given articles, answer the given question. Think step by step. Output the executing result of every step.',
                f'Question:\n{q}\n\nCorpus:\n{ctx}',256,True)
            answer = call(cloud, 'According to the given question and reasoning steps, output the complete final answer as short as possible. If the question is yes-no, output only yes or no.',
                f'----------Question----------\n{q}\n\n----------Steps----------\n{reasoning}',64)
        else:
            collaborative = method == 'caliper' or (method == 'caliper_clf' and branch == 'collaborative')
            if collaborative:
                guidance = call(cloud,'You are an expert. Given a question, output step-by-step instructions.',q,64)
                steps = '\n'.join(f'Step {i+1}: {s}' for i,s in enumerate(guidance.splitlines()))
                prompt = f'Question: {q}\n\nReference: {ctx}\n\nThe following steps are hints from an expert (for reference only):\n{steps}\n\nReason and answer, but do not be limited by the hints.'
            else:
                prompt = f'Question: {q}\n\nReference: {ctx}\n\nThink step by step and answer.'
            reasoning = call(local,'You are a helpful assistant.',prompt,256)
            middle = 'Given the reasoning process:' if collaborative else 'Reasoning:'
            answer = call(local,'You are a helpful assistant.',f'Question: {q}\n\n{middle}\n{reasoning}\n\nExtract and output the final answer concisely. Output only the answer itself, no extra words.',64)
        return {'prediction':answer,'reasoning':reasoning,'guidance':guidance,'calls':calls}

    summary = {'status':'COMPLETED_CANDIDATE_NOT_FROZEN','dataset':args.dataset,'methods':{}}
    for method in METHODS:
        for phase, rows in [('warmup',sample[:args.warmup]),('measured',sample)]:
            def one(i,row):
                start = time.perf_counter()
                try:
                    result = measure_one(row['question'],method,retrieve,generate)
                    return dict(result,index=i,question=row['question'],error=None)
                except Exception as e:
                    # Do not persist response bodies, request headers, or secrets.
                    return {'index':i,'question':row['question'],'seconds':time.perf_counter()-start,
                            'error':type(e).__name__,'status_code':getattr(e,'status_code',None)}
            records=[]
            with (dest/f'{method}_{phase}.jsonl').open('x') as out, ThreadPoolExecutor(max_workers=4) as executor:
                futures=[executor.submit(one,i,r) for i,r in enumerate(rows)]
                for future in as_completed(futures):
                    result=future.result(); records.append(result)
                    out.write(json.dumps(result,ensure_ascii=False)+'\n'); out.flush()
                    if len(records)%20==0 or result['error']:
                        print('PROGRESS',args.dataset,method,phase,len(records),'errors',sum(r['error'] is not None for r in records),flush=True)
            if any(r['error'] for r in records):
                raise RuntimeError(f'{method} {phase}: errors recorded; no automatic retry or promotion')
            if phase=='measured':
                values=[r['seconds'] for r in records]
                summary['methods'][method]={'n':len(values),'mean':float(np.mean(values)),
                    'p50':float(np.percentile(values,50)),'p95':float(np.percentile(values,95))}
        print('METHOD_DONE',args.dataset,method,flush=True)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['hotpotqa','musique','2wikimultihopqa'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=6020)
    p.add_argument('--limit',type=int,default=200)
    p.add_argument('--warmup',type=int,default=20)
    main(p.parse_args())
