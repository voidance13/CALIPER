#!/usr/bin/env python3
"""Assemble a self-contained CALIPER companion analysis archive."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/data/caliper/code/CALIPER")
OUT = ROOT / "companion_archive_20260918"
ROUTED = Path("/tmp/true_routed_text_grouped_20260918")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(src: Path, rel: str, role: str, records: list) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    records.append({"path": rel, "size_bytes": dst.stat().st_size,
                    "sha256": sha256(dst), "role": role,
                    "source": str(src)})


def main() -> None:
    if OUT.exists():
        unexpected = [path for path in OUT.iterdir() if path.name != "paper_snapshot"]
        if unexpected:
            raise SystemExit(f"refusing to overwrite existing archive directory: {OUT}")
    else:
        OUT.mkdir(parents=True)
    records = []

    # Paper snapshot from the synchronized Overleaf checkout is copied by the
    # caller into OUT/paper_snapshot before this script runs.
    if not (OUT / "paper_snapshot/cas-dc-template.tex").exists():
        raise SystemExit("paper snapshot missing; copy the synchronized source first")
    for path in (OUT / "paper_snapshot").rglob("*"):
        if path.is_file():
            records.append({"path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size,
                            "sha256": sha256(path), "role": "synchronized paper source"})

    # Strict text-grouped source-router artifact and its split provenance.
    for src in (ROOT / "query_clf/text_grouped_source_router_20260917").rglob("*"):
        if src.is_file():
            copy_file(src, str(Path("source_router") / src.relative_to(ROOT / "query_clf/text_grouped_source_router_20260917")),
                      "strict text-grouped source-router artifact", records)

    # Execution classifier and diagnostics used by Tables 6--7.
    for src, rel, role in [
        (ROOT / "query_clf/train_new/9b_clf.bin", "execution_classifier/9b_clf.bin", "frozen execution classifier binary"),
        (ROOT / "query_clf/train_new/9b_valid.txt", "execution_classifier/9b_valid.txt", "execution-classifier validation input"),
        (ROOT / "query_clf/stopwords.txt", "execution_classifier/stopwords.txt", "shared preprocessing"),
        (ROOT / "classifier_analysis/classifier_threshold_0997_results.json", "classifier_analysis/classifier_threshold_0997_results.json", "frozen threshold results"),
        (ROOT / "classifier_analysis/classifier_threshold_0997_B32_results.json", "classifier_analysis/classifier_threshold_0997_B32_results.json", "frozen threshold results"),
        (ROOT / "classifier_analysis/classifier_downstream_groups.json", "classifier_analysis/classifier_downstream_groups.json", "Table 6--7 0.997 confusion groups"),
    ]:
        copy_file(src, rel, role, records)

    # New-router Table 8 retrieval records and the rerouting provenance.
    for src, rel, role in [
        (ROUTED / "merged_records.jsonl", "retrieval_table8/逐题召回_records.jsonl", "24,274-question routed retrieval records"),
        (ROUTED / "summary.json", "retrieval_table8/summary.json", "authoritative Table 8 summary"),
        (ROUTED / "reroute_plan.json", "retrieval_table8/reroute_plan.json", "old/new route reconciliation"),
    ]:
        copy_file(src, rel, role, records)

    # Aggregate results and paired-bootstrap outputs cited by the manuscript.
    result_files = [
        "table-result/table2_paper_final_metrics.json",
        "table-result/table3_full_unified_metrics.json",
        "table-result/ablation_final_metrics.json",
        "table-result/table4_quality_latency_ci.json",
        "table-result/table5_summary.json",
        "table-result/table6_sentence_recall_final.json",
        "table-result/bootstrap_ci_results.json",
    ]
    result_files += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "significance_analysis").glob("*.json"))]
    for rel in result_files:
        copy_file(ROOT / rel, str(Path("results") / rel), "aggregate or paired-bootstrap result", records)

    # Include the exact prediction inputs referenced by the result JSONs.
    refs = set()
    for rel in result_files:
        try:
            obj = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        except Exception:
            continue
        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in {"path", "source"} and isinstance(child, str):
                        refs.add(child)
                    elif key == "sources" and isinstance(child, list):
                        refs.update(x for x in child if isinstance(x, str))
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(obj)
    for rel in sorted(refs):
        src = ROOT / rel
        if src.exists() and src.is_file():
            copy_file(src, str(Path("predictions") / rel), "prediction file referenced by an aggregate result", records)

    # Baseline runs needed to reconstruct the classifier behavior labels.
    for src in sorted((ROOT / "query_clf").glob("*_base_sample[123].json")):
        copy_file(src, str(Path("classifier_inputs") / src.name), "baseline run used for difficulty labels", records)

    # Relevant scripts, including the strict router and rerouting implementation.
    scripts = [
        "work/repair/build_text_grouped_source_router.py",
        "work/repair/audit_source_router_text_independence.py",
        "work/repair/true_routed_recall_heldout.py",
        "work/repair/run_text_grouped_reroute.py",
        "scripts/classifier_downstream_groups.py",
        "scripts/classifier_threshold_pareto.py",
        "scripts/paired_bootstrap_caliper.py",
        "scripts/paired_bootstrap_ablation2403.py",
        "scripts/paired_bootstrap_adaptive.py",
        "scripts/paired_bootstrap_table3_ablation.py",
        "scripts/eval_table2_final.py",
        "scripts/eval_table2_musique_paired.py",
        "scripts/eval_table3_full.py",
    ]
    for rel in scripts:
        src = ROOT / rel
        if src.exists():
            copy_file(src, str(Path("scripts") / rel), "reproduction or audit script", records)

    summary = json.loads((ROUTED / "summary.json").read_text(encoding="utf-8"))
    readme = f"""# CALIPER companion analysis archive

Generated: {datetime.now(timezone.utc).isoformat()}

This archive records the frozen artifacts and deterministic outputs used for the current manuscript revision. Dataset source files are not redistributed; the public benchmark inputs are identified by the scripts and result metadata.

## Current authoritative changes

- Source router: strict text-grouped fastText artifact in `source_router/`.
- Source-router official evaluation: 24,274 questions; accuracy {summary['router_accuracy']:.2f}%; zero normalized-text overlap between train/validation/holdout and official evaluation sets.
- Table 8 routed retrieval (new router): HotpotQA {summary['by_dataset']['hp']['R@3']:.2f}/{summary['by_dataset']['hp']['R@5']:.2f}/{summary['by_dataset']['hp']['R@10']:.2f}; MuSiQue {summary['by_dataset']['ms']['R@3']:.2f}/{summary['by_dataset']['ms']['R@5']:.2f}/{summary['by_dataset']['ms']['R@10']:.2f}; 2Wiki {summary['by_dataset']['tw']['R@3']:.2f}/{summary['by_dataset']['tw']['R@5']:.2f}/{summary['by_dataset']['tw']['R@10']:.2f}.
- Table 6 and Table 7 classifier diagnostics use the single frozen execution threshold 0.997; their per-question groups are in `classifier_analysis/classifier_downstream_groups.json`.
- Every JSONL record is keyed by dataset and original row ID; the final Table 8 record set contains {summary['total_records']} unique records and zero retrieval errors.

## Reproducibility notes

The source-router binary and execution-classifier binary are included with SHA-256 entries in `manifest.json`. `retrieval_table8/summary.json` is the authoritative new-router Table 8 summary; `retrieval_table8/逐题召回_records.jsonl` contains the corresponding per-question recall records. Aggregate F1 and latency results are copied under `results/`, and the paired-bootstrap JSON files are retained unchanged. Scripts are included under `scripts/` with their original relative paths.

The original fastText command and hyperparameter log for the execution classifier were not available; the manuscript therefore makes an artifact-based reproducibility statement for that component.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "archive_version": "2026-09-18-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_snapshot": "synchronized Overleaf source",
        "table8_summary": summary,
        "files": sorted(records, key=lambda row: row["path"]),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"archive": str(OUT), "file_count": len(records),
                      "bytes": sum(row["size_bytes"] for row in records),
                      "table8_records": summary["total_records"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
