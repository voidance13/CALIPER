"""Frozen V0/V1/V2 generation with audited query-only inputs and four GPU queues."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import queue
import random
import shutil
import socket
import subprocess
import threading
import time
import urllib.request

BASE = Path('/mnt/data/caliper/code/CALIPER')
RUN = BASE / 'router_e2e_runs/queryonly_repair_20260924_v1'
PY = '/home/gpt/anaconda3/envs/qwen35/bin/python'
PORTS = (6110, 6111, 6112, 6113)
EXPECTED = {'hotpotqa': 7405, 'musique': 2403, '2wikimultihopqa': 12576}
MODEL = '/mnt/data/caliper/models/Qwen3.5-9B'
LOCK = threading.Lock()
TLS = threading.local()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules():
    import sys
    sys.path.insert(0, str(RUN / 'source'))
    core = load_module('router_e2e_rerun_20260918', RUN / 'source/router_e2e_rerun_20260918.py')
    sys.modules['router_e2e_rerun_20260918'] = core
    base = load_module('base_branch', RUN / 'source/router_e2e_v2v4_rerun_20260919.py')
    return core, base


def optimize(module, removed, return_removed):
    tree = ast.parse(Path(module.__file__).read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'run_one')
    fn.body = [x for x in fn.body if not (isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id in removed for t in x.targets))]
    ret = next(x for x in fn.body if isinstance(x, ast.Return))
    pairs = [(k, v) for k, v in zip(ret.value.keys, ret.value.values) if k.value not in return_removed]
    ret.value.keys, ret.value.values = map(list, zip(*pairs))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), module.__file__, 'exec'), module.__dict__)


def selfcheck(core, base):
    originals = (core.run_one, base.run_one)
    optimize(core, {'r3', 'v3'}, {'v3'})
    optimize(base, {'v4_reason', 'v4_pred'}, {'v4'})
    item = {'dataset': 'hotpotqa', 'row_id': 0, 'question': 'Which person?', 'routed': {'retrieved_text': ['Evidence.'], 'predicted_route': 'hotpotqa', 'model': 'specialist'}}
    contexts = {'hotpotqa': {'Which person?': ['Base evidence.']}}
    for hint in ('1. Identify the person in the evidence.', ''):
        steps = {'hotpotqa': {'Which person?': hint}}
        for mode in ('correct', 'revise', 'giveup'):
            calls = []
            def fake(client, prompt, max_tokens=128, system='You are a helpful assistant.'):
                calls.append((prompt, max_tokens, system))
                if system == core.REFLECT_EVID:
                    return 'JUDGMENT: Correct' if mode == 'correct' else 'JUDGMENT: Incorrect\nPLAN: Check evidence.'
                if 'JUDGMENT: Incorrect' in prompt and mode == 'giveup':
                    return 'cannot answer'
                return 'stable response'
            core.call = base.call = fake
            old = originals[0](item, None, contexts, steps)
            expected_calls = calls[:2] + calls[4:]
            calls.clear()
            new = core.run_one(item, None, contexts, steps)
            assert new == {k: old[k] for k in ('v0', 'v1')} and calls == expected_calls
            calls.clear()
            old = originals[1](item, None, contexts, steps)
            expected_calls = calls[:2]
            calls.clear()
            new = base.run_one(item, None, contexts, steps)
            assert new == {'v2': old['v2']} and calls == expected_calls
    print('SELF_CHECK_PASS: prompts, token limits, reflection and missing-guidance fallback preserved', flush=True)


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    assert not (RUN / 'inputs.jsonl').exists(), 'Prepared input already exists; use --run'
    (RUN / 'source').mkdir(exist_ok=True)
    sources = ['router_e2e_rerun_20260918.py', 'router_e2e_v2v4_rerun_20260919.py', 'router_e2e_musique_unified_bgebase_20260920.py', 'finalize_router_e2e_20260918.py']
    for name in sources:
        shutil.copy2(BASE / 'work' / name, RUN / 'source' / name)
    shutil.copy2(__file__, RUN / 'source' / Path(__file__).name)
    core, branch = modules()
    files = {str(RUN / 'source' / n): sha(RUN / 'source' / n) for n in sources}
    def track(p):
        p = Path(p)
        files[str(p)] = sha(p)
        return p
    def lines(p):
        return [json.loads(s) for s in track(p).open() if s.strip()]
    oldroute = BASE / 'router_e2e_runs/text_grouped_20260918/routed_records.jsonl'
    newroute = BASE / 'router_e2e_runs/text_grouped_20260920_unified_bgebase/routed_records.jsonl'
    items = core.load_items(track(oldroute))
    replacements = {r['question']: r for r in lines(newroute) if core.DATASET_ALIASES.get(r['dataset'], r['dataset']) == 'musique'}
    base = {d: {r['question']: core.text_list(r) for r in read(track(BASE / f'data/{d}/test_retrieved.json'))} for d in EXPECTED}
    support = {'musique': {r['question']: core.text_list(r) for r in read(track(BASE / 'data/musique/test_retrieved_unified_bgebase_base.json'))}}
    config = {'hotpotqa': ('batch_hotpotqa_caliper/hotpotqa_caliper_steps.jsonl', 'batch_hotpotqa_caliper/hotpotqa_caliper_steps_result.jsonl'),
              'musique': ('batch_qonly_instr/musique_instr.jsonl', 'batch_qonly_instr/instr_musique_result.jsonl'),
              '2wikimultihopqa': ('batch_qonly_instr/2wikimultihopqa_instr.jsonl', 'batch_qonly_instr/instr_2wikimultihopqa_result.jsonl')}
    maps = {}
    with gzip.open(RUN / 'guidance_provenance.jsonl.gz', 'wt', encoding='utf-8') as f:
        for d, (request_file, result_file) in config.items():
            results = {r['custom_id']: r for r in lines(BASE / result_file)}
            maps[d] = {}
            for r in lines(BASE / request_file):
                users = [m['content'] for m in r['body']['messages'] if m['role'] == 'user']
                assert len(users) == 1 and isinstance(users[0], str)
                q = users[0]
                response = results.get(r['custom_id'], {}).get('response', {}).get('body', {})
                choices = response.get('choices') or []
                text = (choices[0]['message'].get('content') or '') if choices else ''
                maps[d].setdefault(q, []).append({'text': text, 'custom_id': r['custom_id'], 'request_file': request_file, 'result_file': result_file})
                f.write(json.dumps({'dataset': d, 'question': q, 'custom_id': r['custom_id'], 'request': r['body'], 'response': response}, ensure_ascii=False) + '\n')
    missing, flags, counts = [], [], Counter()
    with (RUN / 'inputs.jsonl').open('w', encoding='utf-8') as f:
        for item in items:
            d, q = item['dataset'], item['question']
            assert q in maps[d], (d, 'no exact question-only request', q)
            assert len(maps[d][q]) == 1, (d, 'ambiguous guidance for evaluation question', q)
            g = maps[d][q][0]
            if not g['text'].strip():
                missing.append({'dataset': d, 'row_id': item['row_id'], 'question': q, 'custom_id': g['custom_id']})
            if any(s in g['text'].lower() for s in ('answer:', 'reasoning:', 'the answer is')):
                flags.append({'dataset': d, 'row_id': item['row_id'], 'text': g['text']})
            if d == 'musique':
                item['routed'] = replacements[q]
            assert q in base[d] and base[d][q], (d, 'missing BASE context', q)
            item['core_context'] = core.context_for(item, support if d == 'musique' else base)
            item['v2_context'] = branch.base_context(item, base)
            item['routed'] = {k: item['routed'].get(k) for k in ('predicted_route', 'model')}
            item['guidance'] = g['text']
            item['guidance_origin'] = {k: v for k, v in g.items() if k != 'text'}
            item['input_sha256'] = hashlib.sha256(json.dumps(item, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            counts[d] += 1
    assert dict(counts) == EXPECTED
    dump(RUN / 'missing_guidance.json', missing)
    dump(RUN / 'guidance_structure_flags.json', flags)
    assert len(missing) <= 10, f'Unexpected missing guidance count: {len(missing)}'
    assert not flags, 'Inspect flagged non-procedural content before launching'
    selfcheck(core, branch)
    for n in ['inputs.jsonl', 'guidance_provenance.jsonl.gz', 'source/' + Path(__file__).name]:
        files[str(RUN / n)] = sha(RUN / n)
    dump(RUN / 'manifest.json', {'created_at': now(), 'status': 'PREPARED_NOT_FROZEN', 'datasets': dict(counts), 'input_sha256': files,
         'model': MODEL, 'temperature': 0, 'seed': 42, 'thinking': False, 'token_limits': [256, 64, 160],
         'variants': ['v0', 'v1', 'v2'], 'missing_guidance_policy': 'retained original no-hint fallback; questions not dropped',
         'missing_guidance_count': len(missing), 'v2_context_policy': 'frozen September 19 BASE top-five sources retained',
         'core_context_policy': 'September 18 routed HP/2Wiki; September 20 MuSiQue routed + unified BASE top-three',
         'cloud_calls': 0, 'optimization_selfcheck': 'PASS', 'evaluation_status': 'not yet generated'})
    print('PREPARED', dict(counts), 'missing_guidance', len(missing), flush=True)


def start_servers():
    owned = []
    for port in PORTS:
        with socket.socket() as probe:
            probe.bind(('0.0.0.0', port))
    for gpu, port in enumerate(PORTS):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=2)
        except Exception:
            pass
        else:
            raise RuntimeError(f'Port {port} already occupied; inspect owner')
        log = (RUN / f'vllm_gpu{gpu}.log').open('a')
        args = [PY, '-m', 'vllm.entrypoints.openai.api_server', '--model', MODEL, '--port', str(port), '--trust-remote-code',
                '--max-model-len', '8192', '--max-num-seqs', '64', '--max-num-batched-tokens', '8192', '--enable-prefix-caching', '--no-enable-log-requests']
        proc = subprocess.Popen(args, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        owned.append({'gpu': gpu, 'port': port, 'pid': proc.pid, 'argv': args, 'proc': proc})
    dump(RUN / 'servers.json', [{k: v for k, v in r.items() if k != 'proc'} for r in owned])
    deadline = time.monotonic() + 1200
    waiting = set(PORTS)
    while waiting and time.monotonic() < deadline:
        for r in owned:
            if r['proc'].poll() is not None:
                raise RuntimeError(f'Server GPU {r["gpu"]} exited; inspect log')
        for port in list(waiting):
            try:
                obj = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=2))
                assert any(m['id'] == MODEL for m in obj['data'])
                waiting.remove(port)
            except Exception:
                pass
        if waiting:
            time.sleep(5)
    assert not waiting, f'Servers not ready: {waiting}'
    return owned


def run():
    manifest = read(RUN / 'manifest.json')
    for p, expected in manifest['input_sha256'].items():
        if str(RUN) in p:
            assert sha(p) == expected, ('frozen run input changed', p)
    core, base = modules()
    selfcheck(core, base)
    core.context_for = lambda item, unused: item['core_context']
    base.base_context = lambda item, unused: item['v2_context']
    state = {'status': 'STARTING_SERVERS', 'started_at': now(), 'pid': os.getpid(), 'completed_jobs': 0, 'total_jobs': 2 * sum(EXPECTED.values()), 'errors': 0}
    dump(RUN / 'status.json', state)
    servers = start_servers()
    clients = [core.OpenAI(base_url=f'http://127.0.0.1:{p}/v1', api_key='EMPTY', timeout=300, max_retries=0) for p in PORTS]
    trace = gzip.open(RUN / 'local_requests.jsonl.gz', 'at', encoding='utf-8')
    results = (RUN / 'jobs.jsonl').open('a', encoding='utf-8')
    total_tokens = 0

    def call(client, prompt, max_tokens=128, system='You are a helpful assistant.'):
        nonlocal total_tokens
        TLS.stage += 1
        request = {'model': MODEL, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}],
                   'temperature': 0.0, 'seed': 42, 'max_tokens': max_tokens, 'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
        last = ''
        for attempt in range(3):
            t0 = time.monotonic()
            record = {'job_id': TLS.job, 'port': TLS.port, 'stage': TLS.stage, 'attempt': attempt + 1, 'started_at': now(), 'request': request}
            try:
                response = client.chat.completions.create(**request)
                record['response'] = response.model_dump()
                text = response.choices[0].message.content or ''
            except Exception as e:
                record['error'] = last = repr(e)
            record['seconds'] = time.monotonic() - t0
            with LOCK:
                trace.write(json.dumps(record, ensure_ascii=False) + '\n')
                trace.flush()
                if 'response' in record:
                    total_tokens += (record['response'].get('usage') or {}).get('completion_tokens', 0)
            if 'response' in record:
                return text
            time.sleep(2)
        raise RuntimeError(last)

    core.call = base.call = call
    rows = [json.loads(s) for s in (RUN / 'inputs.jsonl').open() if s.strip()]
    jobs = [(r, variant) for r in rows for variant in ('core', 'base')]
    random.Random(42).shuffle(jobs)
    done = set()
    if (RUN / 'jobs.jsonl').stat().st_size:
        for s in (RUN / 'jobs.jsonl').open():
            r = json.loads(s)
            if 'error' not in r:
                assert r['job_id'] not in done
                done.add(r['job_id'])
    jobs = [(r, v) for r, v in jobs if f'{r["dataset"]}:{r["row_id"]}:{v}' not in done]
    state['completed_jobs'] = len(done)
    t_run = time.monotonic()

    def phase(part, workers, label):
        nonlocal state
        pending = queue.Queue()
        for job in part:
            pending.put(job)
        start, tok0, count0 = time.monotonic(), total_tokens, state['completed_jobs']
        state.update(status=label, workers_per_gpu=workers, updated_at=now())
        dump(RUN / 'status.json', state)
        def worker(gpu):
            while True:
                try:
                    row, variant = pending.get_nowait()
                except queue.Empty:
                    return
                TLS.job = f'{row["dataset"]}:{row["row_id"]}:{variant}'
                TLS.stage, TLS.port = 0, PORTS[gpu]
                out = {'job_id': TLS.job, 'dataset': row['dataset'], 'row_id': row['row_id'], 'question': row['question'],
                       'input_sha256': row['input_sha256'], 'port': PORTS[gpu], 'variant_group': variant}
                t0 = time.monotonic()
                try:
                    fn = core.run_one if variant == 'core' else base.run_one
                    out['predictions'] = fn(row, clients[gpu], {}, {row['dataset']: {row['question']: row['guidance']}})
                except Exception as e:
                    out['error'] = repr(e)
                out['seconds'] = time.monotonic() - t0
                with LOCK:
                    results.write(json.dumps(out, ensure_ascii=False) + '\n')
                    results.flush()
                    if 'error' in out:
                        state['errors'] += 1
                    else:
                        state['completed_jobs'] += 1
                    state.update(updated_at=now(), elapsed_seconds=round(time.monotonic() - t_run, 1))
                    if (state['completed_jobs'] + state['errors']) % 20 == 0:
                        dump(RUN / 'status.json', state)
                        print(json.dumps(state), flush=True)
        with ThreadPoolExecutor(max_workers=workers * 4) as pool:
            fs = [pool.submit(worker, g) for g in range(4) for _ in range(workers)]
            for future in fs:
                future.result()
        elapsed = time.monotonic() - start
        point = {'phase': label, 'workers_per_gpu': workers, 'jobs': state['completed_jobs'] - count0, 'seconds': elapsed,
                 'completion_tokens_per_second': (total_tokens - tok0) / elapsed, 'errors_total': state['errors']}
        print('PHASE', json.dumps(point), flush=True)
        assert state['errors'] == 0, 'Generation errors recorded; inspect before continuing'
        return point

    tune, offset = [], 0
    for workers in (8, 16, 24):
        tune.append(phase(jobs[offset:offset + 192], workers, f'TUNING_{workers}'))
        offset += 192
        dump(RUN / 'throughput_tuning.json', tune)
        if workers == 16 and tune[-1]['completion_tokens_per_second'] < tune[0]['completion_tokens_per_second'] * 1.05:
            break
    best = max(tune, key=lambda x: x['completion_tokens_per_second'])['workers_per_gpu']
    phase(jobs[offset:], best, 'RUNNING_FULL')
    results.close()
    trace.close()
    assert state['completed_jobs'] == state['total_jobs']
    output = {d: {v: [] for v in ('v0', 'v1', 'v2')} for d in EXPECTED}
    seen = set()
    for s in (RUN / 'jobs.jsonl').open():
        r = json.loads(s)
        if 'error' in r:
            continue
        assert r['job_id'] not in seen
        seen.add(r['job_id'])
        for v, pred in r['predictions'].items():
            output[r['dataset']][v].append(pred)
    (RUN / 'predictions').mkdir(exist_ok=True)
    for d in EXPECTED:
        for v, predictions in output[d].items():
            assert len(predictions) == EXPECTED[d]
            dump(RUN / 'predictions' / f'{d}_{v}_queryonly_20260924.json', sorted(predictions, key=lambda r: r['row_id']))
    state.update(status='GENERATION_COMPLETE_AWAITING_STATISTICS', finished_at=now(), selected_workers_per_gpu=best)
    dump(RUN / 'status.json', state)
    for server in servers:
        server['proc'].terminate()
    print('GENERATION_COMPLETE', json.dumps(state), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        try:
            run()
        except Exception as e:
            dump(RUN / 'failure.json', {'at': now(), 'error': repr(e)})
            raise
