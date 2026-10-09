#!/usr/bin/env python3
"""Assemble a versioned routed-record snapshot for the MuSiQue E2E rerun."""
import hashlib
import json
from pathlib import Path

ROOT = Path("/mnt/data/caliper/code/CALIPER")
OLD = ROOT / "router_e2e_runs/text_grouped_20260918/routed_records.jsonl"
NEW_DIR = Path("/tmp/musique_test_unified_bgebase_20260920")
OUT = ROOT / "router_e2e_runs/text_grouped_20260920_unified_bgebase"


def canon(s):
    return " ".join("".join(c if c.isalnum() else " " for c in str(s).lower()).split())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    test = json.loads((ROOT / "data/musique/test.json").read_text(encoding="utf-8"))
    old = []
    for line in OLD.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            old.append(row)
    new = []
    for path in sorted(NEW_DIR.glob("ms_shard_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                new.append(json.loads(line))
    if len(new) != 1523:
        raise RuntimeError(f"new MuSiQue routed rows changed: {len(new)}")
    if len({r["row_id"] for r in new}) != len(new):
        raise RuntimeError("duplicate new MuSiQue row_ids")
    new_questions = {r["question"] for r in new}
    old = [
        row for row in old
        if not (row.get("dataset") == "ms" and row.get("question") in new_questions)
    ]
    out_rows = old + new
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "routed_records.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "input_sha256.txt").write_text(f"{sha(out)}  {out.name}\n", encoding="utf-8")
    meta = {
        "protocol": "frozen text-grouped source router; MuSiQue test specialist branch replaced with unified bge-base retrieval; HP/2Wiki and misrouted MuSiQue evidence reused from 20260918 snapshot",
        "test_rows": len(test),
        "new_musique_specialist_rows": len(new),
        "old_rows_retained": len(old),
        "total_rows": len(out_rows),
        "output": str(out),
        "sha256": sha(out),
    }
    (OUT / "assembly_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
