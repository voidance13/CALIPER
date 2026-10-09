"""Merge new MuSiQue specialist-route records into the frozen routed audit."""
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path("/mnt/data/caliper/code/CALIPER")
OLD = Path("/tmp/true_routed_text_grouped_20260918/final_records/all_records.jsonl")
NEW_DIR = Path("/tmp/true_routed_unified_bgebase_20260920")
ROUTER = ROOT / "query_clf/text_grouped_source_router_20260917/domain_clf.bin"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


old_rows = load_jsonl(OLD)
new_rows = []
for path in sorted(NEW_DIR.glob("ms_to_ms_s*.jsonl")):
    new_rows.extend(load_jsonl(path))
old_ms = {(row["dataset"], row["row_id"]): row for row in old_rows if row["dataset"] == "ms" and row["predicted_route"] == "ms"}
new_ms = {(row["dataset"], row["row_id"]): row for row in new_rows}
assert len(new_ms) == len(new_rows) == 2862
assert set(new_ms) == set(old_ms)
assert all(row.get("error") is None and row.get("recall") for row in new_rows)

merged = []
for row in old_rows:
    key = (row["dataset"], row["row_id"])
    merged.append(new_ms.get(key, row))
assert len(merged) == len(old_rows)
assert len({(row["dataset"], row["row_id"]) for row in merged}) == len(merged)

def aggregate(rows):
    values = [row for row in rows if row["recall"] is not None]
    return {
        "n": len(values),
        **{f"R@{k}": round(statistics.mean(row["recall"][f"R@{k}"] for row in values) * 100, 2) for k in (3, 5, 10)},
    }


new_ms_all = [row for row in merged if row["dataset"] == "ms"]
old_ms_all = [row for row in old_rows if row["dataset"] == "ms"]
by_route = {route: aggregate([row for row in new_ms_all if row["predicted_route"] == route]) for route in ("hp", "ms", "tw")}
new_result = aggregate(new_ms_all)
old_result = aggregate(old_ms_all)
old_router_summary = json.loads((Path("/tmp/true_routed_text_grouped_20260918") / "summary.json").read_text(encoding="utf-8"))
by_dataset_new_router = {
    "hotpotqa": old_router_summary["by_dataset"]["hp"],
    "musique": new_result,
    "2wikimultihopqa": old_router_summary["by_dataset"]["tw"],
}
mixed_macro = {
    f"R@{k}": round(statistics.mean(by_dataset_new_router[name][f"R@{k}"] for name in by_dataset_new_router), 2)
    for k in (3, 5, 10)
}
output = {
    "definition": "MuSiQue official-dev CALIPER source-router retrieval with only the MuSiQue specialist branch replaced by unified bge-base models",
    "gold": "official supporting paragraphs",
    "n": len(new_ms_all),
    "protocol": {"raw_rows": 4834, "evaluated_rows": 4293, "skipped": 541},
    "router": str(ROUTER),
    "router_sha256": sha256(ROUTER),
    "new_specialist_model": {
        "embedder": str(ROOT / "retrieve_fine_tuning/models/unified_bgebase_musique_firstpos_embedder"),
        "reranker": str(ROOT / "retrieve_fine_tuning/models/unified_large_ftsep_musique"),
    },
    "route_counts": dict(Counter(row["predicted_route"] for row in new_ms_all)),
    "new": new_result,
    "old": old_result,
    "delta_new_minus_old": {key: round(new_result[key] - old_result[key], 2) for key in ("R@3", "R@5", "R@10")},
    "by_predicted_route": by_route,
    "all_domain_new_router": by_dataset_new_router,
    "mixed_macro_new_router": mixed_macro,
    "old_source": str(OLD),
    "new_shards": [str(path) for path in sorted(NEW_DIR.glob("ms_to_ms_s*.jsonl"))],
}
(ROOT / "table-result/table6_musique_unified_bgebase_routed.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
(NEW_DIR / "final_records").mkdir(exist_ok=True)
(NEW_DIR / "final_records/all_records.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged), encoding="utf-8"
)
print(json.dumps(output, ensure_ascii=False, indent=2))
