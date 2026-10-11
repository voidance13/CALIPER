"""Dependency-free static smoke test for the public CALIPER checkout."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "configs/current_protocol.json").read_text(encoding="utf-8"))
    assert config["models"]["local_generation"] == "Qwen3.5-9B"
    assert config["models"]["cloud_reference"] == "Qwen3.7-plus"
    assert set(config["execution_thresholds"]) == {"hotpotqa", "musique", "2wikimultihopqa"}
    for relative in (
        "hybrid_predict_separate.py",
        "hybrid_predict_separate_clf.py",
        "normal_predict_separate.py",
        "build_retrieval_cache.py",
        "scripts/eval_table2_final.py",
    ):
        ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
    print("CALIPER public static smoke test: PASS")


if __name__ == "__main__":
    main()
