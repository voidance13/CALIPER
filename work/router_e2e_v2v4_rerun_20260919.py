"""Scheme-A V2/V4 ablation rerun using the frozen question set and base retrieval."""
import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from router_e2e_rerun_20260918 import (
    BASE, MODEL, DATASETS, WORKERS, OpenAI, call, load_contexts,
    load_items, load_steps,
)

RUN = BASE / "router_e2e_runs/v2v4_text_grouped_20260919"
INPUT = BASE / "router_e2e_runs/text_grouped_20260918/routed_records.jsonl"
INSTR2 = "Extract and output the final answer concisely. Output only the answer itself, no extra words."
V4_SYS1 = "You are a helpful assistant. According to given articles, answer the given question. Think step by step. Output the executing result of every step."
V4_SYS2 = 'According to the given question and the reasoning steps, output the complete final answer as short as possible. Specially, if the given question is a yes-no question, just output "yes" or "no".'


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def base_context(item, base):
    vals = base[item["dataset"]].get(item["question"], [])
    return "\n".join(f"[{i+1}] {s}" for i, s in enumerate(vals[:5]))


def run_one(item, client, base, steps):
    q, d = item["question"], item["dataset"]
    ctx = base_context(item, base)
    st = steps[d].get(q, "")
    if st:
        st_txt = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(st.splitlines()))
        p2 = f"Question: {q}\n\nReference: {ctx}\n\nThe following steps are hints from an expert (for reference only):\n{st_txt}\n\nReason and answer, but do not be limited by the hints."
    else:
        p2 = f"Question: {q}\n\nReference: {ctx}\n\nThink step by step and answer."
    v2_reason = call(client, p2, 256)
    v2_pred = call(client, f"Question: {q}\n\nGiven the reasoning process:\n{v2_reason}\n\n{INSTR2}", 64)

    v4_reason = call(client, f"Question:\n{q}\n\nCorpus:\n{ctx}", 256, system=V4_SYS1)
    v4_pred = call(client, f"----------Question----------\n{q}\n\n----------Steps----------\n{v4_reason}", 64, system=V4_SYS2)
    common = {"dataset": d, "row_id": item["row_id"], "question": q, "route": "BASE", "model": MODEL}
    return {"v2": dict(common, pred=v2_pred), "v4": dict(common, pred=v4_pred)}


def run(args):
    RUN.mkdir(parents=True, exist_ok=True)
    snapshot = RUN / "routed_records.jsonl"
    if not snapshot.exists():
        shutil.copyfile(INPUT, snapshot)
        (RUN / "input_sha256.txt").write_text(f"{sha256(snapshot)}  {snapshot.name}\n", encoding="utf-8")
    items = load_items(snapshot)
    base = load_contexts(items)
    steps = load_steps()
    selected = [x for i, x in enumerate(items) if i % args.nshards == args.shard]
    out = RUN / f"shard_{args.shard}.jsonl"
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["question"])
    selected = [x for x in selected if x["question"] not in done]
    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")
    print(f"START shard={args.shard}/{args.nshards} n={len(selected)} port={args.port}", flush=True)
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(run_one, x, client, base, steps) for x in selected]
        for i, fut in enumerate(as_completed(futures), 1):
            f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"shard={args.shard} {i}/{len(selected)} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"DONE shard={args.shard} elapsed={time.time()-t0:.0f}s -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--nshards", type=int, default=4)
    args = p.parse_args()
    if not 0 <= args.shard < args.nshards:
        raise SystemExit("bad shard")
    run(args)


if __name__ == "__main__":
    main()
