"""Scheme-A routed end-to-end rerun for CALIPER Tables 2/4.

This script writes only to the versioned router_e2e_runs directory.  It runs
one mutually exclusive global shard and emits JSONL records for V1, V0, V3.
"""
import argparse
import hashlib
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

BASE = Path("/mnt/data/caliper/code/CALIPER")
RUN = BASE / "router_e2e_runs/text_grouped_20260918"
MODEL = "/mnt/data/caliper/models/Qwen3.5-9B"
DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")
DATASET_ALIASES = {"hp": "hotpotqa", "ms": "musique", "tw": "2wikimultihopqa"}
N_SHARDS = 4
WORKERS = 8

STEP_SRC = {
    "hotpotqa": "hybrid_pred_7b_72b_sep_clf_t2caliper_think.json",
    "musique": "hybrid_pred_7b_72b_sep_use_best_hit_merge_full.json",
    "2wikimultihopqa": "hybrid_pred_7b_72b_sep_use_best_hit_merge_full.json",
}

INSTR2 = "Extract and output the final answer concisely. Output only the answer itself, no extra words."
REFLECT_EVID = (
    "You are a strict answer verifier with EVIDENCE REQUIREMENT. You will be given reasoning steps, corpus, question, and a previous answer. "
    'CRITICAL RULE: You may output "JUDGMENT: Correct" ONLY IF you can QUOTE the exact corpus sentence that supports the answer. '
    "Steps:\n1. Find the corpus sentence(s) that directly answer the question.\n"
    "2. If the previous answer matches the quoted sentence's key entity/fact, output:\n"
    '   JUDGMENT: Correct\n   EVIDENCE: "<exact quote>"\n'
    "3. If you CANNOT find a supporting sentence, or the answer is vague/incomplete/missing-info, output:\n"
    "   JUDGMENT: Incorrect\n   REASON: <one sentence>\n   PLAN: <new concise plan>\n"
)
GIVEUP = re.compile(r"(cannot|can not|not possible|does not (specify|contain|provide)|unable to|no information|not found)", re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical(s: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in str(s).lower()).split())


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def text_list(row):
    return row.get("retrieved_text") or [st for c in row.get("ctxs", []) for st in c.get("sentences", [])]


def load_mu_keys():
    """Freeze the already-used 2,403-question Table-2 intersection."""
    test = load_json(BASE / "data/musique/test.json")
    sources = [
        "data/musique/musique_baseline_2stage_soft.json",
        "baseline_runs/naive_rag_musique_full_seed42_w1.json",
        "adaptive_runs/adaptive_musique_full_seed42_v3.json",
        "always_multi_runs/always_multi_ircot_musique_full_seed42.json",
        "data/musique/musique_clf_t2_th0997.json",
        "data/musique/musique_v1_ubh_2stream.json",
        "data/musique/direct_pred_t2_qwen37_A.json",
    ]
    counts = [Counter(canonical(r.get("question", "")) for r in load_json(BASE / p)) for p in sources]
    test_counts = Counter(canonical(r["question"]) for r in test)
    keys = [canonical(r["question"]) for r in test if test_counts[canonical(r["question"])] == 1 and all(c[k] == 1 for c in counts for k in [canonical(r["question"])])]
    if len(keys) != 2403:
        raise RuntimeError(f"frozen MuSiQue intersection changed: {len(keys)}")
    return set(keys)


def load_items(records_path: Path):
    routed = {d: {} for d in DATASETS}
    for line in records_path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            row["dataset"] = DATASET_ALIASES.get(row["dataset"], row["dataset"])
            routed[row["dataset"]][row["question"]] = row
    mu_keys = load_mu_keys()
    items = []
    for d in DATASETS:
        tests = load_json(BASE / f"data/{d}/test.json")
        for i, t in enumerate(tests):
            q = t["question"]
            if d == "musique" and canonical(q) not in mu_keys:
                continue
            if q not in routed[d]:
                raise RuntimeError(f"missing routed record: {d} row={i}")
            items.append({"dataset": d, "row_id": i, "question": q, "routed": routed[d][q]})
    expected = 7405 + 2403 + 12576
    if len(items) != expected:
        raise RuntimeError(f"unexpected task count {len(items)} != {expected}")
    return items


def load_contexts(items):
    base = {}
    for d in DATASETS:
        p = BASE / f"data/{d}/test_retrieved.json"
        if p.exists():
            base[d] = {r["question"]: text_list(r) for r in load_json(p)}
        else:
            base[d] = {}
    return base


def load_steps():
    out = {d: {} for d in DATASETS}
    for d in DATASETS:
        p = BASE / "data" / d / STEP_SRC[d]
        if not p.exists():
            continue
        for row in load_json(p):
            st = row.get("res1") or row.get("steps") or ""
            if isinstance(st, str) and st.strip():
                out[d][row["question"]] = st
    return out


def call(client, prompt, max_tokens=128, system="You are a helpful assistant."):
    last = ""
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.0,
                seed=42,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            last = repr(e)
            time.sleep(2)
    raise RuntimeError(last or "empty model response")


def context_for(item, base):
    d, q = item["dataset"], item["question"]
    ev = (item["routed"].get("retrieved_text") or [])[:5]
    if d != "musique":
        return "\n".join(f"[{i+1}] {s}" for i, s in enumerate(ev))
    ans = base[d].get(q, [])[:3]
    seen, lines = set(), []
    for s in ev + ans:
        k = re.sub(r"\s+", " ", s.strip().lower())
        if k and k not in seen:
            seen.add(k)
            lines.append(s)
    return "\n".join(f"[{i+1}] {s}" for i, s in enumerate(lines))


def run_one(item, client, base, steps):
    q, d = item["question"], item["dataset"]
    ctx = context_for(item, base)
    st = steps[d].get(q, "")
    if st:
        st_txt = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(st.splitlines()))
        p1 = f"Question: {q}\n\nReference: {ctx}\n\nThe following steps are hints from an expert (for reference only):\n{st_txt}\n\nReason and answer, but do not be limited by the hints."
    else:
        p1 = f"Question: {q}\n\nReference: {ctx}\n\nThink step by step and answer."
    r1 = call(client, p1, 256)
    v1 = call(client, f"Question: {q}\n\nGiven the reasoning process:\n{r1}\n\n{INSTR2}", 64)
    # V3 removes cloud steps and reflection, while retaining the same routed context.
    r3 = call(client, f"Question: {q}\n\nReference: {ctx}\n\nThink step by step and answer.", 256)
    v3 = call(client, f"Question: {q}\n\nReasoning:\n{r3}\n\n{INSTR2}", 64)
    best, cur = v1, v1
    for _ in range(3):
        rr = call(client, f"----------Steps----------\n{r1}\n\n----------Corpus----------\n{ctx}\n\n----------Question----------\n{q}\n\n----------Previous answer----------\n{cur}", 160, REFLECT_EVID)
        if rr.strip().startswith("JUDGMENT: Correct"):
            break
        cur = call(client, f"----------Steps----------\n{r1}\n\n{rr}\n\n----------Question----------\n{q}\n\n{INSTR2}", 64)
        if GIVEUP.search(cur):
            cur = best
            break
    common = {"dataset": d, "row_id": item["row_id"], "question": q, "pred": v1, "route": item["routed"].get("predicted_route"), "model": item["routed"].get("model")}
    return {"v1": dict(common), "v0": dict(common, pred=cur), "v3": dict(common, pred=v3)}


def run(args):
    RUN.mkdir(parents=True, exist_ok=True)
    records_path = RUN / "routed_records.jsonl"
    if not records_path.exists():
        raise RuntimeError(f"missing copied routed records: {records_path}")
    items = load_items(records_path)
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
            result = fut.result()
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"shard={args.shard} {i}/{len(selected)} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"DONE shard={args.shard} elapsed={time.time()-t0:.0f}s -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--nshards", type=int, default=N_SHARDS)
    args = p.parse_args()
    if not 0 <= args.shard < args.nshards:
        raise SystemExit("bad shard")
    run(args)


if __name__ == "__main__":
    main()
