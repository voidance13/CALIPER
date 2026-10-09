"""Authorized new four-question annotation repair, not historical reproduction.

Uses current paper model/settings, retained two-stage Table 3 prompt forms,
fresh query-only guidance, one output per stage, and no automatic API retry.
Starts and stops only its own local inference server. Originals never change.
"""
import argparse
import ast
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI
import httpx

BASE = Path('/mnt/data/caliper/code/CALIPER')
ANNOTATION = BASE / 'annotation_repairs/2wiki_sf_20261002'
MODEL = '/mnt/data/caliper/models/Qwen3.5-9B'
CLOUD = 'qwen3.7-plus'
REUSE = BASE / 'annotation_repairs/2wiki_sf_generation_20261004_v2/requests'
LOCAL_SYS = 'You are a helpful assistant. According to given articles, answer the given question. Think step by step. Output the executing result of every step.'
LOCAL_EXTRACT = 'According to the given question and the reasoning steps, output the complete final answer as short as possible. Specially, if the given question is a yes-no question, just output "yes" or "no".'
CORE_SYS = 'You are a helpful assistant. According to the given steps and corpus, answer the given question. Output the executing result of every step in simple sentences.'
CORE_EXTRACT = 'According to the given question and reasoning steps, output the final answer as short as possible.'
GUIDANCE_SYS = "You are a helpful assistant. Output the solution steps for the given question. Don't use any information that is not given. Don't give any example. Only generate necessary steps succinctly."


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def credential():
    value = os.environ.get('ALIYUN_API_KEY') or dotenv_values(BASE / '.env').get('ALIYUN_API_KEY')
    if not value:
        raise RuntimeError('Configured cloud credential unavailable')
    return value


def assert_prompt_sources():
    # Verify that prompt text comes from retained experiment source, rather
    # than assuming an unstated historical invocation/configuration.
    normal = (BASE / 'normal_predict_separate.py').read_text()
    hybrid = (BASE / 'hybrid_predict_separate.py').read_text()
    constants = lambda source: {n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {LOCAL_SYS, LOCAL_EXTRACT} <= constants(normal)
    assert {CORE_SYS, CORE_EXTRACT, GUIDANCE_SYS} <= constants(hybrid)


def request(client, output, qid, method, stage, messages, tokens, cloud):
    extra = {'enable_thinking': False} if cloud else {'chat_template_kwargs': {'enable_thinking': False}}
    body = {'model': CLOUD if cloud else MODEL, 'messages': messages,
            'temperature': 0.0, 'seed': 42, 'max_tokens': tokens, 'extra_body': extra}
    if stage == 'guidance':
        assert cloud and messages == [{'role': 'system', 'content': GUIDANCE_SYS}, {'role': 'user', 'content': messages[1]['content']}]
        assert 'Corpus:' not in messages[1]['content']
    stem = f'{qid}_{method}_{stage}'
    request_path = output / f'{stem}.request.json'
    response_path = output / f'{stem}.response.json'
    # Retain every already-completed stage exactly; do not resample a stage
    # because the previous save operation failed before its next request.
    hits = []
    for previous in REUSE.glob('*.request.json'):
        old_request = load(previous)
        old_response = previous.with_name(previous.name.replace('.request.json', '.response.json'))
        if old_request['body'] == body and old_response.exists():
            hits.append((previous, old_response, old_request, load(old_response)))
    assert len(hits) <= 1, 'Ambiguous cached stage; do not select best output'
    if hits:
        previous, previous_response, old_request, old_response = hits[0]
        value = old_response['response']['choices'][0]['message']['content']
        assert value and value.strip()
        write(request_path, old_request)
        write(response_path, old_response)
        write(output / f'{stem}.reuse.json', {'original_request': str(previous), 'request_sha256': sha(previous),
                                            'original_response': str(previous_response), 'response_sha256': sha(previous_response),
                                            'body_exact_match': True, 'no_new_api_call': True})
        return value
    write(request_path, {'created_utc': timestamp(), 'body': body,
                                            'cloud_role': 'question_only_guidance' if stage == 'guidance' else ('evidence_grounded_reference' if cloud else 'none')})
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(**body)
        write(response_path, {'created_utc': timestamp(), 'elapsed_seconds': time.perf_counter() - start,
                                                'response': response.model_dump(mode='json')})
        value = response.choices[0].message.content
        if not value or not value.strip():
            raise RuntimeError('Empty response')
        return value
    except Exception as error:
        # Do not log exception bodies, headers, authorization or credentials.
        write(output / f'{stem}.failure.json', {'created_utc': timestamp(), 'error_type': type(error).__name__,
                                                'http_status': getattr(error, 'status_code', None), 'automatic_retry': False})
        raise RuntimeError(f'{qid} {method}/{stage} failed: {type(error).__name__}') from None


def corpus(row):
    context = {c['title']: c['sentences'] for c in row['ctxs']}
    return '\n'.join(context[t][i] for t, i in row['supporting_facts'])


def run_question(row, output, local, cloud):
    q, qid, text = row['question'], row['id'], corpus(row)
    def call(method, stage, system, user, tokens=256, remote=False):
        return request(cloud if remote else local, output, qid, method, stage,
                       [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], tokens, remote)
    guidance = call('CALIPER', 'guidance', GUIDANCE_SYS, q, remote=True)
    common = {'question': q, 'answer': row['answers'][0], 'id': qid}
    records = {}
    for method, remote in [('Qwen3.5-9B', False), ('Qwen3.7-plus', True)]:
        reasoning = call(method, 'reasoning', LOCAL_SYS, 'Question:\n' + q + '\n\nCorpus:\n' + text, remote=remote)
        answer = call(method, 'extraction', LOCAL_EXTRACT,
                      '----------Question----------\n' + q + '\n\n----------Steps----------\n' + reasoning, 64, remote)
        records[method] = common | {'res1': reasoning, 'pred': answer}
    reasoning = call('CALIPER', 'reasoning', CORE_SYS,
                     '----------Question----------\n' + q + '----------Corpus----------\n' + text +
                     '\n\n----------Steps----------\n' + guidance + '\n\n----------Question----------\n' + q)
    answer = call('CALIPER', 'extraction', CORE_EXTRACT,
                  '----------Steps----------\n' + reasoning + '\n\n----------Question----------\n' + q, 64)
    records['CALIPER'] = common | {'steps': guidance, 'res1': reasoning, 'pred': answer,
                                  'reflection': '', 'reflection_iterations': 0}
    write(output / f'{qid}_repaired_predictions.json', records)
    print('COMPLETED_QUESTION', qid, flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    rows = load(ANNOTATION / 'four_corrected_questions.json')
    assert len(rows) == 4 and len({r['id'] for r in rows}) == 4
    manifest = load(ANNOTATION / 'repair_manifest.json')
    for name, expected in manifest['output_sha256'].items():
        assert sha(ANNOTATION / name) == expected
    assert_prompt_sources()
    key = credential()
    with socket.socket() as check:
        assert check.connect_ex(('127.0.0.1', 6110)) != 0, 'Port occupied; do not reuse an unverified server'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'requests').mkdir()
    inputs = [ANNOTATION / 'four_corrected_questions.json', ANNOTATION / 'repair_manifest.json',
              BASE / 'normal_predict_separate.py', BASE / 'hybrid_predict_separate.py', Path(__file__).resolve(),
              Path(MODEL) / 'config.json', Path(MODEL) / 'generation_config.json']
    command = ['/home/gpt/anaconda3/envs/qwen35/bin/python', '-m', 'vllm.entrypoints.openai.api_server',
               '--model', MODEL, '--served-model-name', MODEL, '--host', '127.0.0.1', '--port', '6110',
               '--seed', '42', '--gpu-memory-utilization', '0.65', '--max-model-len', '8192']
    recipe = {'created_utc': timestamp(), 'status': 'NEW_AUTHORIZED_REPAIR_RUN',
              'not_historical_reproduction': True, 'consent': 'User approved a new current-configuration four-question repair on 2026-10-04.',
              'n': 4, 'models': {'local': MODEL, 'cloud': CLOUD}, 'temperature': 0.0, 'seed': 42,
              'thinking_enabled': False, 'stage_token_limits': {'reasoning': 256, 'extraction': 64, 'guidance': 256},
              'reflection_enabled': False, 'prompt_scope': 'Retained two-stage fully relevant-evidence Table 3 prompt forms with explicitly current model/decoding settings.',
              'original_invocation_and_model_binding': 'NOT_VERIFIED',
              'automatic_api_retries': 0, 'question_concurrency': 4, 'server_cuda_visible_devices': '0',
              'server_command': command, 'input_sha256': {str(p): sha(p) for p in inputs if p.exists()},
              'cloud_reference_receives_corrected_supporting_sentences': True,
              'caliper_cloud_guidance_receives_question_only': True,
              'classifier_threshold_unchanged': 0.977, 'classifier_composition': 'All four route to CALIPER; no independent classifier generation.',
              'command': sys.argv,
              'local_http_proxy_bypassed': True,
              'preceding_preflight': '2wiki_sf_generation_20261004: localhost proxy returned HTTP 502; stopped before any generation request.',
              'stage_continuation': 'v2 retained four guidance and four local reasoning outputs; save-name collision prevented the following extraction requests. Reuse exact-matching request bodies/responses and execute only unfinished stages.',
              'retained_stage_input_sha256': {str(p): sha(p) for p in REUSE.glob('*.json')}}
    write(output / 'run_recipe.json', recipe)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
    start = time.monotonic()
    process = None
    log = (output / 'local_server.log').open('x')
    try:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(output / 'server_process.json', {'pid': process.pid, 'started_utc': timestamp(), 'owned_by_this_run': True})
        print('SERVER_STARTING', process.pid, flush=True)
        local = OpenAI(base_url='http://127.0.0.1:6110/v1', api_key='local-unused', max_retries=0, timeout=120,
                       http_client=httpx.Client(trust_env=False))
        remote = OpenAI(base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', api_key=key, max_retries=0, timeout=120)
        while True:
            if process.poll() is not None:
                raise RuntimeError('Owned inference server exited before readiness')
            if time.monotonic() - start > 600:
                raise TimeoutError('Inference server readiness exceeded 600 seconds')
            try:
                model_list = local.models.list().model_dump(mode='json')
                assert MODEL in {m['id'] for m in model_list['data']}
                write(output / 'served_models.json', model_list)
                break
            except Exception:
                time.sleep(2)
        print('SERVER_READY', flush=True)
        generated = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = {pool.submit(run_question, row, output / 'requests', local, remote): row for row in rows}
            for future in as_completed(pending, timeout=900):
                row = pending[future]
                generated[row['question']] = future.result()
        assert len(generated) == 4
        for path, expected in recipe['input_sha256'].items():
            assert sha(Path(path)) == expected
        write(output / 'replacement_predictions.json', generated)
        write(output / 'GENERATION_RECEIPT.json', {'status': 'PASS_NEW_FOUR_QUESTION_GENERATION',
                                                'completed_utc': timestamp(), 'elapsed_seconds': time.monotonic() - start,
                                                'repaired_question_n': 4, 'generation_method_n': 3,
                                                'historical_reproduction': False,
                                                'output_sha256': {str(p.relative_to(output)): sha(p) for p in output.rglob('*.json')}})
        print('GENERATION_PASS', flush=True)
    except Exception as error:
        write(output / 'RUN_FAILURE.json', {'created_utc': timestamp(), 'error_type': type(error).__name__,
                                         'status': 'FAILED_NOT_ADOPTED', 'automatic_retry': False})
        print('GENERATION_FAILED', type(error).__name__, flush=True)
        raise SystemExit(1)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
        log.close()
        write(output / 'SERVER_CLEANUP.json', {'created_utc': timestamp(), 'owned_server_stopped': process is None or process.poll() is not None})


if __name__ == '__main__':
    main()
