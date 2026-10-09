"""Supplement frozen Table 5 estimates using retained, question-paired rows.

No inference, sample selection, timing measurement or raw-file modification.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")
METHODS = ("baseline", "caliper_clf", "caliper", "cloud_deep")
COMPARISONS = (
    ("cloud_deep", "baseline"),
    ("cloud_deep", "caliper"),
    ("cloud_deep", "caliper_clf"),
    ("caliper", "caliper_clf"),
)
FILES = {
    "hotpotqa": "hotpotqa_paired_composition.json",
    "musique": "musique_resample_paired_composition.json",
    "2wikimultihopqa": "2wikimultihopqa_paired_composition.json",
}
REMOTE = "/mnt/data/caliper/code/CALIPER"
CANDIDATE = REMOTE + "/classifier_analysis/overlap_free_retrain_20260927/figure3_table5_candidate_20260929_v2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(values):
    return [float(x) for x in np.percentile(values, (2.5, 97.5))]


def bootstrap(matrix):
    # The question is the resampling unit; all four methods share each draw.
    rng = np.random.default_rng(42)
    draws = np.empty((10000, len(METHODS)))
    for start in range(0, 10000, 100):
        indices = rng.integers(0, len(matrix), size=(100, len(matrix)))
        draws[start:start + 100] = matrix[indices].mean(axis=1)
    return draws


def load_current(inputs, dataset, summary, replacement):
    old_rows = read(inputs / FILES[dataset])
    reference = {method: {} for method in ("baseline", "caliper", "cloud_deep")}
    for row in old_rows:
        method = row["method"]
        if method not in reference:
            continue
        if dataset == "hotpotqa" and row["index"] == 177:
            row = replacement["rows"][method]
        question = row["question"]
        assert question not in reference[method], (dataset, method, "duplicate question")
        assert row.get("error") is None, (dataset, method, "failed row")
        reference[method][question] = row["seconds"]
    clf_rows = [json.loads(line) for line in
                (inputs / f"table5_{dataset}_candidate_rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [row["question"] for row in clf_rows]
    assert len(questions) == len(set(questions)) == 200, (dataset, "classifier row count")
    for method, values in reference.items():
        assert set(values) == set(questions), (dataset, method, "pairing mismatch")
    matrix = np.empty((200, len(METHODS)))
    paired = []
    for i, row in enumerate(clf_rows):
        q = row["question"]
        assert row.get("error") is None
        assert row["threshold"] == summary["thresholds"][dataset]
        assert row["route"] == ("local" if row["p_easy"] >= row["threshold"] else "collaborative")
        # The retained replacement retrieval total already includes classifier time.
        includes_classifier = dataset == "hotpotqa" and row["index"] == 177
        stages = row["generation_seconds"] + row["retrieval_seconds"]
        if includes_classifier:
            adopted = replacement["rows"]["caliper_clf"]
            assert row["question"] == adopted["question"]
            assert row["seconds"] == adopted["seconds"]
            assert row["retrieval_seconds"] == replacement["retrieval"]["routed_clf"]["total_seconds"]
        else:
            stages += row["classifier_seconds"]
        assert abs(row["seconds"] - stages) < 2e-8
        values = {"caliper_clf": row["seconds"], **{m: v[q] for m, v in reference.items()}}
        matrix[i] = [values[m] for m in METHODS]
        paired.append({"dataset": dataset, "question": q, "index": row["index"],
                       "clf_route": row["route"], "clf_already_in_retrieval_total": includes_classifier,
                       "seconds": values})
    assert np.isfinite(matrix).all() and (matrix > 0).all()
    expected = summary["datasets"][dataset]
    diagnostics = {}
    for col, method in enumerate(METHODS):
        values = matrix[:, col]
        target = expected["candidate_caliper_clf"] if method == "caliper_clf" else expected["measured_reference_rows"][method]
        actual = {"n": 200, "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
                  "p95": float(np.percentile(values, 95))}
        assert target["n"] == 200
        assert all(abs(actual[k] - target[k]) < 0.000051 for k in ("mean", "p50", "p95")), (dataset, method, actual, target)
        diagnostics[method] = actual
    return matrix, paired, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Do not overwrite an existing analysis"
    summary = read(args.inputs / "table5_candidate_summary.json")
    replacement = read(args.inputs / "hotpotqa_replacement_3716.json")
    assert sha(args.inputs / "hotpotqa_replacement_3716.json") == summary["inputs"]["replacement_evidence"]["hotpotqa"]["sha256"]
    result = {
        "status": "SUPPLEMENTAL_ANALYSIS_OF_ADOPTED_TABLE5_NOT_ARCHIVE_FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "based_on_archive": "2026-09-30-v8",
        "protocol": {"metric": "estimated end-to-end latency", "unit": "question",
                     "n_per_dataset": 200, "bootstrap_samples": 10000, "seed": 42,
                     "confidence_level": 0.95, "interval": "percentile",
                     "paired_across_methods": True, "new_inference_or_timing": False,
                     "new_exclusions_or_replacements": False, "multiplicity_adjustment": "none; descriptive pointwise intervals",
                     "scope": "Conditional on retained stage timings and the adopted sample; not across-run, network-day or integrated-system uncertainty.",
                     "main_quality_protocol_not_paired_here": True},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "script_sha256": sha(Path(__file__)),
        "inputs": {}, "datasets": {},
    }
    all_rows = []
    for dataset in DATASETS:
        matrix, paired, diagnostics = load_current(args.inputs, dataset, summary, replacement)
        all_rows.extend(paired)
        draws = bootstrap(matrix)
        # Deterministic rerun of analysis, not new observations.
        assert np.array_equal(draws, bootstrap(matrix))
        means = matrix.mean(axis=0)
        for i, method in enumerate(METHODS):
            diagnostics[method]["mean_ci95"] = ci(draws[:, i])
        contrasts = {}
        for reference, candidate in COMPARISONS:
            a, b = METHODS.index(reference), METHODS.index(candidate)
            delta = draws[:, a] - draws[:, b]
            reduction = delta / draws[:, a] * 100
            contrasts[f"{candidate}_vs_{reference}"] = {
                "n": 200, "reference_mean": float(means[a]), "candidate_mean": float(means[b]),
                "delta_seconds_reference_minus_candidate": float(means[a] - means[b]),
                "delta_seconds_ci95": ci(delta),
                "ratio_of_means_reduction_percent": float((means[a] - means[b]) / means[a] * 100),
                "ratio_of_means_reduction_percent_ci95": ci(reduction),
            }
        result["datasets"][dataset] = {"methods": diagnostics, "paired_contrasts": contrasts,
                                       "question_set_sha256": hashlib.sha256(json.dumps(sorted(x["question"] for x in paired), ensure_ascii=False).encode()).hexdigest(),
                                       "verification": "200 unique questions; all four methods aligned; all current mean/p50/p95 cells match"}
    paths = {
        "table5_candidate_summary.json": CANDIDATE + "/table5_candidate_summary.json",
        "hotpotqa_replacement_3716.json": REMOTE + "/latency_runs/table5_repair_20260929_v1/hotpotqa_replacement_3716.json",
        "hotpotqa_paired_composition.json": REMOTE + "/latency_runs/composed_table5_20260922_v1/hotpotqa_paired_composition.json",
        "2wikimultihopqa_paired_composition.json": REMOTE + "/latency_runs/composed_table5_20260922_v1/2wikimultihopqa_paired_composition.json",
        "musique_resample_paired_composition.json": REMOTE + "/latency_runs/musique_resample_seed20260922_v1/analysis/paired_composition.json",
        **{f"table5_{d}_candidate_rows.jsonl": CANDIDATE + f"/table5_{d}_candidate_rows.jsonl" for d in DATASETS},
    }
    for name, remote in paths.items():
        result["inputs"][name] = {"remote_path": remote, "sha256": sha(args.inputs / name),
                                 "bytes": (args.inputs / name).stat().st_size}
    args.output.mkdir(parents=True)
    target = args.output / "table5_current_bootstrap_ci.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows_path = args.output / "table5_current_paired_rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8")
    receipt = {"status": "ANALYSIS_VALIDATED_NOT_ARCHIVE_FROZEN", "table5_values_unchanged": True,
               "deterministic_analysis_replay": True, "new_model_calls": 0,
               "command": " ".join(sys.argv), "created_at_utc": result["created_at_utc"],
               "files": {target.name: sha(target), rows_path.name: sha(rows_path)}}
    (args.output / "VALIDATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["datasets"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
