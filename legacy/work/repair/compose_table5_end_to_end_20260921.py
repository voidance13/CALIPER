#!/usr/bin/env python3
"""Compose Table 5 retrieval-plus-generation latency and paired bootstrap CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seconds(value):
    if isinstance(value, dict):
        return None if value.get("error") is not None else value.get("seconds")
    return value


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def paired_bootstrap(deep: np.ndarray, candidate: np.ndarray, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    differences = np.empty(10000)
    reductions = np.empty(10000)
    for start in range(0, 10000, 100):
        size = min(100, 10000 - start)
        indices = rng.integers(0, len(deep), size=(size, len(deep)))
        deep_means = deep[indices].mean(axis=1)
        candidate_means = candidate[indices].mean(axis=1)
        differences[start:start + size] = deep_means - candidate_means
        reductions[start:start + size] = (deep_means - candidate_means) / deep_means * 100
    per_question = (deep - candidate) / deep * 100
    return {
        "n": len(deep),
        "deep_mean": float(deep.mean()),
        "candidate_mean": float(candidate.mean()),
        "delta_seconds": float((deep - candidate).mean()),
        "delta_seconds_ci95": [float(x) for x in np.percentile(differences, (2.5, 97.5))],
        "ratio_of_means_reduction_percent": float((deep.mean() - candidate.mean()) / deep.mean() * 100),
        "ratio_of_means_reduction_percent_ci95": [float(x) for x in np.percentile(reductions, (2.5, 97.5))],
        "mean_per_question_reduction_percent_diagnostic": float(per_question.mean()),
        "bootstrap_samples": 10000,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    args = parser.parse_args()

    retrieval = load(args.retrieval)["datasets"]
    output = {
        "protocol": {
            "label": "end-to-end latency covering retrieval and generation",
            "composition": "per-question online retrieval time plus corresponding frozen generation time",
            "seed": 42,
            "workers": 4,
            "warmup": 20,
            "retrieval_cache": False,
            "model_loading_included": False,
            "retrieval_scope": "FAISS construction + embedding + reranking",
        },
        "datasets": {},
    }
    bootstrap = {"bootstrap_samples": 10000, "seed": 42, "datasets": {}}

    for dataset in DATASETS:
        retrieval_rows = retrieval[dataset]["per_question"]
        timing_rows = load(args.raw_dir / f"table4_timing_{dataset}_seed42.json")["rows"]
        classifier_rows = load(args.raw_dir / f"table4_clf_actual_{dataset}_seed42.json")["rows"]
        cloud_rows = load(args.raw_dir / f"table4_cloud_deep_{dataset}_full_b256_seed42.json")["rows"]
        if not (len(retrieval_rows) == len(timing_rows) == len(classifier_rows) == len(cloud_rows) == 200):
            raise ValueError(f"unexpected row count for {dataset}")

        composed = {name: [] for name in ("baseline", "caliper_clf", "caliper", "cloud_deep")}
        paired = {name: {} for name in composed}
        for index, (retrieval_row, timing_row, classifier_row, cloud_row) in enumerate(
            zip(retrieval_rows, timing_rows, classifier_rows, cloud_rows)
        ):
            question = retrieval_row["question"]
            if not (question == timing_row["question"] == classifier_row["question"] == cloud_row["question"]):
                raise ValueError(f"question mismatch for {dataset} row {index}")
            retrieval_seconds = float(retrieval_row["seconds"])
            values = {
                "baseline": seconds(timing_row["baseline"]),
                "caliper_clf": None if classifier_row.get("error") is not None else classifier_row.get("seconds"),
                "caliper": seconds(timing_row["caliper"]),
                "cloud_deep": None if cloud_row.get("error") is not None else cloud_row.get("seconds"),
            }
            for name, value in values.items():
                if value is not None:
                    total = float(value) + retrieval_seconds
                    composed[name].append(total)
                    paired[name][index] = total

        conditions = {name: stats(values) for name, values in composed.items()}
        output["datasets"][dataset] = {
            "retrieval": stats([float(row["seconds"]) for row in retrieval_rows]),
            "conditions": conditions,
        }
        common = sorted(set(paired["caliper"]) & set(paired["cloud_deep"]))
        deep = np.asarray([paired["cloud_deep"][i] for i in common], dtype=float)
        caliper = np.asarray([paired["caliper"][i] for i in common], dtype=float)
        bootstrap["datasets"][dataset] = paired_bootstrap(deep, caliper)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.bootstrap.write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
