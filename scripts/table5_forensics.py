import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

import fasttext


ROOT = Path("query_clf")
CASES = {
    "legacy_joint": (
        ROOT / "train/train_bi.bin",
        [ROOT / "train/valid_bi.txt", ROOT / "train/hotpotqa_test2000_easy_bi.txt", ROOT / "train/hotpotqa_test2000_hard_bi.txt", ROOT / "train/musique_test2000_easy_bi.txt", ROOT / "train/musique_test2000_hard_bi.txt"],
    ),
    "legacy_with_2wiki": (
        ROOT / "train/train_bi_2wiki.bin",
        [ROOT / "train/valid_bi_2wiki.txt", ROOT / "train/hotpotqa_test2000_easy_bi.txt", ROOT / "train/hotpotqa_test2000_hard_bi.txt", ROOT / "train/2wikimultihopqa_test2000_easy_bi.txt", ROOT / "train/2wikimultihopqa_test2000_hard_bi.txt"],
    ),
    "legacy_musique_only": (
        ROOT / "train/train_bi_musique_only.bin",
        [ROOT / "train/musique_test2000_easy_bi.txt", ROOT / "train/musique_test2000_hard_bi.txt", ROOT / "train/valid_bi.txt"],
    ),
    "musique_new": (ROOT / "train_new/musique_new.bin", [ROOT / "train_new/valid_musique.txt", ROOT / "train_new/train_musique.txt"]),
    "general_all": (ROOT / "train_new/general_all.bin", [ROOT / "train_new/general_all_valid.txt"]),
    "l1o_hotpotqa": (ROOT / "train_new/l1o_hotpotqa.bin", [ROOT / "train_new/l1o_hotpotqa_valid.txt"]),
    "l1o_musique": (ROOT / "train_new/l1o_musique.bin", [ROOT / "train_new/l1o_musique_valid.txt"]),
    "l1o_2wiki": (ROOT / "train_new/l1o_2wikimultihopqa.bin", [ROOT / "train_new/l1o_2wikimultihopqa_valid.txt"]),
    "9b": (ROOT / "train_new/9b_clf.bin", [ROOT / "train_new/9b_valid.txt"]),
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def metadata(path):
    labels = Counter()
    n = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            n += 1
            labels[line.split(maxsplit=1)[0]] += 1
    return {"n": n, "labels": dict(labels), "sha256": digest(path)}


def main():
    output = {"historical_table5": {"hotpotqa": 0.7186, "musique": 0.6020, "2wikimultihopqa": 0.8688}, "cases": {}}
    for name, (model_path, tests) in CASES.items():
        model = fasttext.load_model(str(model_path))
        row = {"model": str(model_path), "model_sha256": digest(model_path), "tests": {}}
        for test in tests:
            row["tests"][str(test)] = {"accuracy": model.test(str(test))[1], **metadata(test)}
        output["cases"][name] = row
        print(name, {Path(path).name: result["accuracy"] for path, result in row["tests"].items()}, flush=True)
        del model
        gc.collect()
    Path("classifier_analysis").mkdir(exist_ok=True)
    Path("classifier_analysis/table5_forensics.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
