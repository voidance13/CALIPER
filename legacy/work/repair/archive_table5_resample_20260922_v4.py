"""Freeze the adopted MuSiQue Table 5 resample as archive v4 on gpu01."""
import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


D = Path("/mnt/data/caliper/archives/CALIPER_companion_archive_history")
VERSION = "2026-09-22-v4"
NAME = "CALIPER_companion_archive_20260922_v4_CURRENT_FROZEN"
STAGE = D / f"{NAME}.staging"
OUTPUT = D / f"{NAME}.tar.gz"
INPUT = D / "refresh_20260922_v4_inputs"
EVIDENCE = Path(
    "/mnt/data/caliper/code/CALIPER/latency_runs/"
    "musique_resample_seed20260922_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def extract_archive(source: Path, destination: Path) -> None:
    with tarfile.open(source, "r|gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            assert member.isfile()
            relative = Path(*Path(member.name).parts[1:])
            target = destination / relative
            assert not relative.is_absolute()
            assert target.resolve().is_relative_to(destination.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source_stream, target.open("wb") as target_stream:
                shutil.copyfileobj(source_stream, target_stream)


def verify_manifest(root: Path) -> None:
    manifest = read_json(root / "manifest.json")
    for record in manifest["files"]:
        path = root / record["path"]
        assert path.stat().st_size == record["size_bytes"]
        assert sha256(path) == record["sha256"]


def rounded_triplet(record):
    return [round(record[key] + 1e-12, 2) for key in ("mean", "p50", "p95")]


def main() -> None:
    required_inputs = [
        "cas-dc-template.tex",
        "cas-dc-template.pdf",
        "Supplementary_Configuration_Notes.md",
        "paper_commit.txt",
        "TABLE5_COMPOSED_20260922_SUMMARY.json",
        "TABLE5_COMPOSED_20260922_BOOTSTRAP.json",
    ]
    assert all((INPUT / name).is_file() for name in required_inputs)
    assert (EVIDENCE / "SEALED_SAMPLE.json").is_file()
    assert (EVIDENCE / "VALIDATION_RECEIPT.json").is_file()
    assert (EVIDENCE / "analysis/summary.json").is_file()
    assert (EVIDENCE / "analysis/paired_composition.json").is_file()
    assert not STAGE.exists() and not OUTPUT.exists()

    previous = read_json(D / "CURRENT_FROZEN_ARCHIVE.json")
    assert previous["archive_version"] == "2026-09-22-v3"
    previous_archive = Path(previous["archive_path"])
    assert previous_archive.stat().st_size == previous["archive_size_bytes"]
    assert sha256(previous_archive) == previous["archive_sha256"]

    STAGE.mkdir()
    extract_archive(previous_archive, STAGE)
    verify_manifest(STAGE)
    print("PREDECESSOR_VERIFIED", flush=True)

    historical = STAGE / "historical/pre_20260922_v4"
    historical.mkdir()
    for name in [
        "paper_snapshot",
        "manifest.json",
        "FREEZE_STATUS.json",
        "FREEZE_RECEIPT.json",
        "CURRENT_EVIDENCE_INDEX.json",
        "README.md",
    ]:
        source = STAGE / name
        if source.exists():
            shutil.move(str(source), str(historical / name))

    old_table5 = STAGE / "results/table5_end_to_end_20260921"
    if old_table5.exists():
        target = historical / "results/table5_end_to_end_20260921"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_table5), str(target))

    paper = STAGE / "paper_snapshot"
    shutil.copytree(historical / "paper_snapshot", paper)
    for name in [
        "cas-dc-template.tex",
        "cas-dc-template.pdf",
        "Supplementary_Configuration_Notes.md",
    ]:
        shutil.copy2(INPUT / name, paper / name)

    current_evidence = STAGE / "results/table5_musique_resample_20260922_v1"
    shutil.copytree(EVIDENCE, current_evidence)
    assert sum(1 for path in current_evidence.rglob("*") if path.is_file()) == 28

    retained_summary = read_json(INPUT / "TABLE5_COMPOSED_20260922_SUMMARY.json")
    retained_bootstrap = read_json(INPUT / "TABLE5_COMPOSED_20260922_BOOTSTRAP.json")
    musique_summary = read_json(EVIDENCE / "analysis/summary.json")
    validation = read_json(EVIDENCE / "VALIDATION_RECEIPT.json")
    assert musique_summary["status"] == "VALIDATED_CANDIDATE_NOT_FROZEN"
    assert musique_summary["attempted"] == 800
    assert musique_summary["successful"] == 800
    assert not musique_summary["failures"]

    datasets = {
        "hotpotqa": {
            "sample_seed": 42,
            "conditions": retained_summary["datasets"]["hotpotqa"]["conditions"],
            "context_matches": retained_summary["datasets"]["hotpotqa"]["context_matches"],
        },
        "musique": {
            "sample_seed": 20260922,
            "conditions": musique_summary["conditions"],
            "context_matches": {
                "baseline": 200,
                "caliper_clf": 200,
                "caliper": 200,
                "cloud_deep": 200,
            },
        },
        "2wikimultihopqa": {
            "sample_seed": 42,
            "conditions": retained_summary["datasets"]["2wikimultihopqa"]["conditions"],
            "context_matches": retained_summary["datasets"]["2wikimultihopqa"]["context_matches"],
        },
    }
    table5_current = {
        "status": "CURRENT_FROZEN",
        "version": VERSION,
        "protocol": {
            "label": "end-to-end latency estimate",
            "sample_size_per_dataset": 200,
            "composition": "retrieval and generation timed separately and combined per question according to the execution branch",
            "same_questions_across_methods_within_dataset": True,
            "warmup_questions_per_condition": 20,
            "concurrency": 4,
            "model_loading_included": False,
        },
        "datasets": datasets,
        "evidence": {
            "hotpotqa_2wiki_components": "historical/pre_20260922_v4/results/table5_end_to_end_20260921/",
            "musique": "results/table5_musique_resample_20260922_v1/",
        },
    }

    bootstrap = {
        "status": "CURRENT_FROZEN",
        "version": VERSION,
        "bootstrap_samples": 10000,
        "seed": 42,
        "comparison": "CALIPER core versus two-stage cloud deep-reasoning reference",
        "datasets": {
            "hotpotqa": retained_bootstrap["datasets"]["hotpotqa"],
            "musique": musique_summary["paired_bootstrap_vs_cloud"]["caliper"],
            "2wikimultihopqa": retained_bootstrap["datasets"]["2wikimultihopqa"],
        },
    }

    expected_display = {
        "hotpotqa": {
            "baseline": [3.90, 4.36, 4.62],
            "caliper_clf": [5.15, 5.48, 5.86],
            "caliper": [6.04, 5.51, 9.40],
            "cloud_deep": [8.89, 8.77, 10.54],
        },
        "musique": {
            "baseline": [4.95, 5.18, 5.76],
            "caliper_clf": [6.16, 6.25, 7.05],
            "caliper": [6.36, 6.31, 7.05],
            "cloud_deep": [12.32, 12.20, 15.29],
        },
        "2wikimultihopqa": {
            "baseline": [4.11, 4.46, 4.56],
            "caliper_clf": [5.31, 5.52, 5.97],
            "caliper": [5.93, 5.51, 6.79],
            "cloud_deep": [9.49, 9.74, 10.80],
        },
    }
    for dataset, methods in expected_display.items():
        for method, expected in methods.items():
            assert rounded_triplet(datasets[dataset]["conditions"][method]) == expected

    expected_reductions = {
        "hotpotqa": [32.14, 28.26, 35.56],
        "musique": [48.39, 47.35, 49.39],
        "2wikimultihopqa": [37.50, 34.07, 40.40],
    }
    for dataset, expected in expected_reductions.items():
        record = bootstrap["datasets"][dataset]
        actual = [
            round(record["ratio_of_means_reduction_percent"] + 1e-12, 2),
            round(record["ratio_of_means_reduction_percent_ci95"][0] + 1e-12, 2),
            round(record["ratio_of_means_reduction_percent_ci95"][1] + 1e-12, 2),
        ]
        assert actual == expected

    current_results = STAGE / "results/table5_current_20260922_v4"
    write_json(current_results / "table5_current.json", table5_current)
    write_json(current_results / "table5_current_paired_bootstrap.json", bootstrap)
    write_json(current_results / "musique_validation_receipt.json", validation)

    manuscript_text = (paper / "cas-dc-template.tex").read_text(encoding="utf-8")
    supplement_text = (paper / "Supplementary_Configuration_Notes.md").read_text(encoding="utf-8")
    for required in ["32.14\\%, 48.39\\%, and 37.50\\%", "4.95", "6.16", "6.36", "12.32"]:
        assert required in manuscript_text
    assert "historical timing protocol" not in manuscript_text
    assert "archived generation times" not in manuscript_text
    assert "historical timing protocol" not in supplement_text

    now = datetime.now(timezone.utc)
    status = {
        "archive_version": VERSION,
        "status": "CURRENT_FROZEN",
        "is_current_frozen": True,
        "frozen_at_utc": now.isoformat(),
        "frozen_at_local": now.astimezone(timezone(timedelta(hours=8))).isoformat(),
        "supersedes_archive_version": previous["archive_version"],
        "paper_git_commit": (INPUT / "paper_commit.txt").read_text(encoding="utf-8").strip(),
        "paper_sha256": sha256(paper / "cas-dc-template.tex"),
        "pdf_sha256": sha256(paper / "cas-dc-template.pdf"),
        "scope": "Adopt the sealed MuSiQue random-200 Table 5 rerun, refresh latency wording, and bind current Table 5 evidence to the paper snapshot.",
        "verification": "Table values, bootstrap intervals, evidence completeness, paper bindings, manifest, and packaged bytes verified.",
    }
    write_json(STAGE / "FREEZE_STATUS.json", status)

    previous_receipt = read_json(historical / "FREEZE_RECEIPT.json")
    receipt = {
        "archive_version": VERSION,
        "supersedes": previous["archive_version"],
        "adoption": status,
        "table5_musique": {
            "sample_seed": 20260922,
            "eligible_pool": 2403,
            "sample_size": 200,
            "generation_attempted": 800,
            "generation_successful": 800,
            "failures": 0,
            "source_validation_receipt": "results/table5_musique_resample_20260922_v1/VALIDATION_RECEIPT.json",
        },
        "previous_receipt": "historical/pre_20260922_v4/FREEZE_RECEIPT.json",
        "previous_receipt_archive_version": previous_receipt.get("archive_version"),
    }
    write_json(STAGE / "FREEZE_RECEIPT.json", receipt)

    index = read_json(historical / "CURRENT_EVIDENCE_INDEX.json")
    index["Table5_summary"] = "results/table5_current_20260922_v4/table5_current.json"
    index["Table5_statistics"] = "results/table5_current_20260922_v4/table5_current_paired_bootstrap.json"
    index["Table5_MuSiQue_current_evidence"] = "results/table5_musique_resample_20260922_v1/"
    index["Table5_superseded_evidence"] = "historical/pre_20260922_v4/results/table5_end_to_end_20260921/"
    index["Table5_protocol"] = "Fixed 200-question sample per dataset; retrieval and generation timed separately and combined per question according to each execution branch."
    index["documentation_update"] = {
        "version": VERSION,
        "notes": "paper_snapshot/cas-dc-template.tex and Supplementary_Configuration_Notes.md",
        "table5_values_changed": True,
        "muSiQue_sample_seed": 20260922,
    }
    write_json(STAGE / "CURRENT_EVIDENCE_INDEX.json", index)

    (STAGE / "README.md").write_text(
        f"# CALIPER companion archive {VERSION}\n\n"
        "CURRENT_FROZEN. Start with CURRENT_EVIDENCE_INDEX.json and FREEZE_STATUS.json.\n\n"
        "Table 5 reports end-to-end latency estimates on fixed 200-question samples. "
        "Retrieval and generation are timed separately and combined per question according to each execution branch. "
        "The current MuSiQue evidence is a sealed seed-20260922 random sample from the common 2,403-question intersection; "
        "all four methods use the same 200 questions and completed 800/800 measured generation requests. "
        "HotpotQA and 2Wiki retain their adopted components. Model loading is excluded.\n\n"
        "Superseded archive metadata and Table 5 records remain under historical/pre_20260922_v4/.\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), STAGE / Path(__file__).name)

    files = [
        {
            "path": str(path.relative_to(STAGE)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(STAGE.rglob("*"))
        if path.is_file() and path != STAGE / "manifest.json"
    ]
    write_json(STAGE / "manifest.json", {"version": VERSION, "files": files})
    expected = {record["path"]: record for record in files}
    expected["manifest.json"] = {
        "size_bytes": (STAGE / "manifest.json").stat().st_size,
        "sha256": sha256(STAGE / "manifest.json"),
    }

    print(f"PACKAGING {len(files)}", flush=True)
    subprocess.run(
        ["tar", "-I", "pigz -p 4 -1", "-cf", str(OUTPUT), "-C", str(D), STAGE.name],
        check=True,
    )
    seen = set()
    with tarfile.open(OUTPUT, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            relative = str(Path(*Path(member.name).parts[1:]))
            record = expected[relative]
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(chunk)
            assert member.size == record["size_bytes"]
            assert digest.hexdigest() == record["sha256"]
            seen.add(relative)
    assert seen == set(expected)

    pointer = dict(
        status,
        archive_path=str(OUTPUT),
        remote_archive_path=f"gpu01:{OUTPUT}",
        archive_sha256=sha256(OUTPUT),
        archive_size_bytes=OUTPUT.stat().st_size,
        verification="PASS",
        verified_file_count=len(files),
        local_archive_present=False,
    )
    write_json(D / "CURRENT_FROZEN_ARCHIVE.json", pointer)
    write_json(D / "VERIFICATION_20260922_v4.json", pointer)
    (D / f"{OUTPUT.name}.sha256").write_text(
        f"{pointer['archive_sha256']}  {OUTPUT.name}\n", encoding="utf-8"
    )

    predecessor_verification = read_json(D / "VERIFICATION_20260922_v2.json")
    obsolete = Path(predecessor_verification["archive_path"])
    assert predecessor_verification["archive_version"] == "2026-09-22-v2"
    if obsolete.exists():
        assert obsolete.parent.resolve() == D.resolve()
        assert obsolete.stat().st_size == predecessor_verification["archive_size_bytes"]
        assert sha256(obsolete) == predecessor_verification["archive_sha256"]
        obsolete.unlink()
        obsolete.with_name(f"{obsolete.name}.sha256").unlink(missing_ok=True)

    assert STAGE.parent.resolve() == D.resolve()
    shutil.rmtree(STAGE)
    print("VERIFIED_CURRENT", json.dumps(pointer), flush=True)


if __name__ == "__main__":
    main()
