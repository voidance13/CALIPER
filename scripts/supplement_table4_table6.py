"""Derive Table 4 long-tail sensitivity and Table 6 official-gold macro Mixed."""
import json
import statistics
import sys
from pathlib import Path


root = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/caliper/code/CALIPER")


def read(path):
    return json.loads((root / path).read_text(encoding="utf-8"))


clf = read("table-result/table4_clf_actual_hotpotqa_seed42.json")
caliper = read("table-result/table4_timing_hotpotqa_seed42.json")
clf_rows = [row for row in clf["rows"] if row["error"] is None]
clf_seconds = [row["seconds"] for row in clf_rows]
long_rows = [row for row in clf_rows if row["seconds"] > 20]
caliper_seconds = []
for row in caliper["rows"]:
    value = row["caliper"]
    if isinstance(value, dict):
        if value["error"] is None:
            caliper_seconds.append(value["seconds"])
    else:
        caliper_seconds.append(value)

table4 = {
    "definition": "descriptive sensitivity only; primary mean retains all successful requests",
    "long_tail_rule": "successful latency > 20 seconds",
    "caliper_clf": {
        "n": len(clf_seconds),
        "mean_all_successes": round(statistics.mean(clf_seconds), 3),
        "median_all_successes": round(statistics.median(clf_seconds), 3),
        "long_tail_n": len(long_rows),
        "long_tail_seconds": [round(row["seconds"], 3) for row in long_rows],
        "mean_excluding_gt20_sensitivity": round(
            statistics.mean(row["seconds"] for row in clf_rows if row["seconds"] <= 20), 3
        ),
    },
    "caliper": {"n": len(caliper_seconds), "mean": round(statistics.mean(caliper_seconds), 3)},
}
(root / "table-result/table4_hotpot_clf_longtail_sensitivity.json").write_text(
    json.dumps(table4, ensure_ascii=False, indent=2), encoding="utf-8"
)

sentence = read("table-result/table6_sentence_recall_final.json")["datasets"]
musique_files = {
    "BASE": "base",
    "FT-SEP": "ftsep",
    "FT-MER": "ftmer",
}
musique = {
    method: read(f"table-result/table6_musique_paragraph_recall_{suffix}.json")
    for method, suffix in musique_files.items()
}
musique["CALIPER"] = musique["FT-SEP"]
methods = ("BASE", "FT-SEP", "FT-MER", "CALIPER")
ks = (3, 5, 10)
mixed = {}
for method in methods:
    mixed[method] = {
        f"R@{k}": round(
            statistics.mean(
                (
                    sentence["hotpotqa"][method][f"R@{k}"],
                    musique[method][f"R@{k}"],
                    sentence["2wikimultihopqa"][method][f"R@{k}"],
                )
            ),
            2,
        )
        for k in ks
    }

table6 = {
    "definition": "equal-weight macro average of the three current official-gold domain results",
    "warning": "descriptive cross-domain summary; not a query-level router evaluation; gold units are supporting sentences for HotpotQA/2Wiki and supporting paragraphs for MuSiQue",
    "domains": ["hotpotqa", "musique", "2wikimultihopqa"],
    "mixed_macro": mixed,
    "caliper_definition": "dataset-known in-domain retriever; therefore equals FT-SEP in current Table 6",
    "legacy_mixed_excluded": "old Mixed used approximate MuSiQue first-sentence gold and is not comparable",
}
(root / "table-result/table6_mixed_macro_official.json").write_text(
    json.dumps(table6, ensure_ascii=False, indent=2), encoding="utf-8"
)

assert table4["caliper_clf"]["long_tail_n"] == 2
assert mixed["CALIPER"] == mixed["FT-SEP"]
print(json.dumps({"table4": table4, "table6": table6}, ensure_ascii=False, indent=2))
