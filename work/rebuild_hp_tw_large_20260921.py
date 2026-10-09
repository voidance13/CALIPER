"""Four-GPU FT-SEP cache rebuild; preserve the historical retrieval pipeline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/data/caliper/code/CALIPER')
RUN = ROOT / 'retrieval_runs/hp_tw_large_20260921'
MODELS = {
    'hotpotqa': ('hotpotqa_bge-embedder-ft', 'unified_large_ftsep_hotpotqa', 7405),
    '2wikimultihopqa': ('2wikimultihopqa_bge-embedder-ft', 'unified_large_ftsep_2wiki', 12576),
}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def worker(dataset, shard):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(shard)
    sys.path.insert(0, str(ROOT))
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain.retrievers import ContextualCompressionRetriever
    from BCERerank import BCERerank
    e, r, expected = MODELS[dataset]
    emb = HuggingFaceBgeEmbeddings(
        model_name=str(ROOT / 'retrieve_fine_tuning/models' / e),
        model_kwargs={'device': 'cuda:0'}, encode_kwargs={'normalize_embeddings': True},
        query_instruction='Represent this sentence for searching relevant passages:')
    rer = BCERerank(model=str(ROOT / 'retrieve_fine_tuning/models' / r),
                   top_n=10, device='cuda:0', use_fp16=True)
    tests = json.loads((ROOT / f'data/{dataset}/test.json').read_text())
    assert len(tests) == expected
    start = time.time()
    with (RUN / f'{dataset}_s{shard}.jsonl').open('x') as stream:
        for done, i in enumerate(range(shard, len(tests), 4), 1):
            s = tests[i]
            texts = [sent for c in s['ctxs'] for sent in c['sentences']]
            index = FAISS.from_texts(texts=texts, embedding=emb,
                                    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
            retriever = ContextualCompressionRetriever(base_compressor=rer,
                base_retriever=index.as_retriever(search_type='similarity', search_kwargs={'k': 20}))
            # Fail visibly instead of silently publishing empty results.
            docs = retriever.get_relevant_documents(s['question'])
            rt = [d.page_content for d in docs]
            assert rt, (dataset, i, 'empty retrieval')
            row = {'row_index': i, 'question': s['question'],
                   'answers': s.get('answers', s.get('answer', '')), 'id': s.get('id', i),
                   'type': s.get('type', ''), 'level': s.get('level', ''),
                   'supporting_facts': s.get('supporting_facts', []),
                   'ctxs': [{'title': '', 'sentences': rt}], 'retrieved_text': rt}
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            if done % 100 == 0:
                print(dataset, shard, done, 'seconds', round(time.time()-start), flush=True)
    print('SHARD_DONE', dataset, shard, flush=True)

def recall(tests, rows):
    sums = {k: 0.0 for k in (3, 5, 10)}
    assert len(tests) == len(rows)
    for t, r in zip(tests, rows):
        assert t['question'] == r['question']
        contexts = {c['title']: c for c in t['ctxs']}
        gold = [contexts[title]['sentences'][idx] for title, idx in t['supporting_facts']
                if title in contexts and idx < len(contexts[title]['sentences'])]
        assert gold
        for k in sums:
            sums[k] += sum(s in r['retrieved_text'][:k] for s in gold) / len(gold)
    return {f'R@{k}': round(value / len(tests) * 100, 2) for k, value in sums.items()}

def orchestrate():
    RUN.mkdir(parents=True, exist_ok=True)
    manifest = {'status': 'RUNNING_NOT_FROZEN', 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'script_sha256': sha(Path(__file__)), 'datasets': {},
                'protocol': 'Historical LangChain FAISS MAX_INNER_PRODUCT top20, BCERerank top10; only reranker checkpoint changed; four disjoint shards',
                'cloud_calls': 0}
    for d, (e, r, n) in MODELS.items():
        artifacts = [ROOT / f'data/{d}/test.json', ROOT / f'data/{d}/test_retrieved_embedderft_rerankerft_v2.json']
        for model in (e, r):
            directory = ROOT / 'retrieve_fine_tuning/models' / model
            weights = list(directory.glob('*.safetensors')) + list(directory.glob('pytorch_model*.bin'))
            assert weights, directory
            artifacts.extend(weights + [directory / 'config.json'])
        manifest['datasets'][d] = {'n': n, 'embedder': e, 'reranker': r,
                                   'sha256': {str(p): sha(p) for p in artifacts}}
    with (RUN / 'manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2)
    results = {}
    for dataset, (_, _, n) in MODELS.items():
        processes, logs = [], []
        for gpu in range(4):
            log = (RUN / f'{dataset}_s{gpu}.log').open('x')
            logs.append(log)
            env = dict(os.environ, OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--dataset', dataset, '--shard', str(gpu)]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env))
        deadline = time.time() + 8*3600
        while any(p.poll() is None for p in processes):
            failures = [p.returncode for p in processes if p.poll() not in (None, 0)]
            if failures:
                raise RuntimeError(f'Worker failed: {failures}; no auto-retry; other workers left untouched')
            if time.time() > deadline:
                for p in processes:
                    if p.poll() is None:
                        p.terminate()
                raise TimeoutError('Eight-hour dataset timeout')
            time.sleep(10)
        assert all(p.returncode == 0 for p in processes)
        for log in logs:
            log.close()
        rows = [json.loads(line) for g in range(4)
                for line in (RUN / f'{dataset}_s{g}.jsonl').read_text().splitlines()]
        rows.sort(key=lambda r: r['row_index'])
        assert [r['row_index'] for r in rows] == list(range(n))
        tests = json.loads((ROOT / f'data/{dataset}/test.json').read_text())
        old = json.loads((ROOT / f'data/{dataset}/test_retrieved_embedderft_rerankerft_v2.json').read_text())
        destination = RUN / f'{dataset}_test_retrieved_ftsep_large.json'
        with destination.open('x') as f:
            json.dump(rows, f, ensure_ascii=False)
        results[dataset] = {'n': n, 'old': recall(tests, old), 'new': recall(tests, rows),
                            'cache': str(destination), 'sha256': sha(destination)}
        (RUN / 'results.json').write_text(json.dumps(results, indent=2))
        print('DATASET_DONE', json.dumps(results[dataset]), flush=True)
    manifest['status'] = 'COMPLETED_CANDIDATE_NOT_FROZEN'
    manifest['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    (RUN / 'completion.json').write_text(json.dumps(manifest, indent=2))
    print('ALL_DONE', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=MODELS)
    parser.add_argument('--shard', type=int, choices=range(4))
    args = parser.parse_args()
    worker(args.dataset, args.shard) if args.dataset else orchestrate()
