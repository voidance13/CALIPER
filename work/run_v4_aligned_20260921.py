"""V4 with the exact frozen V3 generation code; only context is BASE top-5."""
import ast
import concurrent.futures as cf
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

BASE = Path('/mnt/data/caliper/code/CALIPER')
RUN = BASE / 'router_e2e_runs/v4_v3aligned_20260921'
SRC = BASE / 'work/router_e2e_rerun_20260918.py'
PY = '/home/gpt/anaconda3/envs/qwen35/bin/python'
spec = importlib.util.spec_from_file_location('frozen', SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def prepare():
    RUN.mkdir(exist_ok=True)
    # Extract the actual V3 assignments, not a reworded prompt implementation.
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_one')
    nodes = [n for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('r3', 'v3') for t in n.targets)]
    assert len(nodes) == 2
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(SRC), 'exec')
    items = m.load_items(BASE / 'router_e2e_runs/text_grouped_20260918/routed_records.jsonl')
    manifest = {'status': 'PREPARED', 'source': str(SRC), 'source_sha256': m.sha256(SRC), 'model': m.MODEL,
                'temperature': 0, 'seed': 42, 'thinking': False, 'max_tokens': [256,64],
                'workers_per_gpu': 8, 'gpu_count': 4, 'cloud_calls': 0,
                'generation_code': ast.unparse(ast.Module(body=nodes, type_ignores=[])), 'datasets': {}}
    for d, expected in zip(m.DATASETS, (7405,2403,12576)):
        path = BASE / f'data/{d}/test_retrieved.json'
        if d == 'musique':
            path = BASE / 'data/musique/test_retrieved_unified_bgebase_base.json'
        cache = {r['question']: m.text_list(r) for r in m.load_json(path)}
        rows = []
        for item in items:
            if item['dataset'] != d:
                continue
            q = item['question']
            assert q in cache and cache[q], (d,q)
            rows.append({'dataset': d, 'row_id': item['row_id'], 'question': q,
                         'context': '\n'.join(f'[{i+1}] {s}' for i,s in enumerate(cache[q][:5]))})
        assert len(rows) == expected and len({r['question'] for r in rows}) == expected
        dst = RUN / f'{d}_inputs.json'
        dst.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
        manifest['datasets'][d] = {'n': len(rows), 'cache': str(path), 'cache_sha256': m.sha256(path), 'input_sha256': m.sha256(dst)}
    (RUN / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return code

def shard(d, g, code):
    rows = m.load_json(RUN / f'{d}_inputs.json')[g::4]
    out = RUN / f'{d}_shard_{g}.jsonl'
    done = {json.loads(s)['question'] for s in out.read_text().splitlines()} if out.exists() else set()
    client = m.OpenAI(base_url=f'http://127.0.0.1:{6010+g}/v1', api_key='EMPTY')
    def one(row):
        env = {'q': row['question'], 'ctx': row['context'], 'client': client, 'call': m.call, 'INSTR2': m.INSTR2}
        exec(code, env)
        return {k:row[k] for k in ('dataset','row_id','question')} | {'pred': env['v3'], 'reasoning': env['r3'], 'model':m.MODEL, 'route':'BASE'}
    with out.open('a', encoding='utf-8') as f, cf.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one,r) for r in rows if r['question'] not in done]
        for i, future in enumerate(cf.as_completed(futures),1):
            f.write(json.dumps(future.result(), ensure_ascii=False)+'\n')
            f.flush()
            if i % 100 == 0:
                print(d,g,i,len(futures),flush=True)
    assert len(out.read_text().splitlines()) == len(rows)

def main():
    code = prepare()
    print('PREFLIGHT_PASS 7405/2403/12576; exact V3 code; BASE contexts', flush=True)
    if '--prepare-only' in __import__('sys').argv:
        return
    procs = []
    try:
        for g in range(4):
            log = (RUN / f'vllm_{g}.log').open('a')
            procs.append(subprocess.Popen([PY,'-m','vllm.entrypoints.openai.api_server','--model',m.MODEL,'--port',str(6010+g),'--trust-remote-code','--max-model-len','8192'], env=os.environ | {'CUDA_VISIBLE_DEVICES':str(g)}, stdout=log, stderr=subprocess.STDOUT))
        for g,p in enumerate(procs):
            deadline = time.time()+900
            while time.time()<deadline:
                if p.poll() is not None:
                    raise RuntimeError(f'server {g} exited')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{6010+g}/v1/models', timeout=3):
                        break
                except Exception:
                    time.sleep(3)
            else:
                raise TimeoutError(f'server {g}')
        for d in m.DATASETS:
            start=time.time()
            print('DATASET_START',d,flush=True)
            with cf.ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(shard,d,g,code) for g in range(4)]
                for f in futures:
                    f.result()
            print('DATASET_DONE',d,'seconds',time.time()-start,flush=True)
        (RUN / 'GENERATION_COMPLETE.json').write_text(json.dumps({'status':'GENERATION_COMPLETE_NOT_YET_EVALUATED','time':time.time()}))
        print('ALL_GENERATION_DONE',flush=True)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()

if __name__ == '__main__':
    main()
