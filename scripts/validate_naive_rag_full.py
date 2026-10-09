import json
import math
import sys
from pathlib import Path


BASE = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])
DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


report = {}
for dataset in DATASETS:
    tests = json.load(
        (BASE / "data" / dataset / "test.json").open(encoding="utf-8")
    )
    result_path = BASE / "baseline_runs" / f"naive_rag_{dataset}_full_seed42_w1.json"
    rows = json.load(result_path.open(encoding="utf-8"))
    latencies = [row["latency_seconds"] for row in rows]
    report[dataset] = {
        "expected_rows": len(tests),
        "actual_rows": len(rows),
        "index_mismatches": sum(row["index"] != index for index, row in enumerate(rows)),
        "question_mismatches": sum(
            row["question"] != test["question"] for row, test in zip(rows, tests)
        ),
        "empty_predictions": sum(not row["pred"].strip() for row in rows),
        "errors": sum(bool(row["error"]) for row in rows),
        "unique_questions": len({row["question"] for row in rows}),
        "latency_mean": round(sum(latencies) / len(latencies), 4),
        "latency_p50": round(percentile(latencies, 0.50), 4),
        "latency_p95": round(percentile(latencies, 0.95), 4),
        "latency_p99": round(percentile(latencies, 0.99), 4),
        "result_file": str(result_path),
    }
    checks = report[dataset]
    if any(
        (
            checks["expected_rows"] != checks["actual_rows"],
            checks["index_mismatches"],
            checks["question_mismatches"],
            checks["empty_predictions"],
            checks["errors"],
        )
    ):
        raise RuntimeError(f"integrity failure: {dataset}: {checks}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", encoding="utf-8") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
