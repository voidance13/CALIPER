import json
from pathlib import Path

import fasttext


root = Path("query_clf")
tests = {}
for dataset in ("hotpotqa", "musique", "2wikimultihopqa"):
    combined = Path(f"/tmp/{dataset}_test_bi.txt")
    with combined.open("w") as target:
        for level in ("easy", "hard"):
            with (root / f"train/{dataset}_test2000_{level}_bi.txt").open() as source:
                target.writelines(source)
    tests[dataset] = combined

models = [
    root / "train/train_bi.bin",
    root / "train/train_bi_2wiki.bin",
    root / "train_new/9b_clf.bin",
    root / "train_new/hotpotqa_new2.bin",
    root / "train_new/musique_new2.bin",
    root / "train_new/2wikimultihopqa_new2.bin",
    root / "train_new/general_all.bin",
]
results = {}
for path in models:
    model = fasttext.load_model(str(path))
    results[path.name] = {name: model.test(str(test))[1] for name, test in tests.items()}
    print(path.name, results[path.name], flush=True)
Path("classifier_analysis").mkdir(exist_ok=True)
Path("classifier_analysis/model_probe.json").write_text(json.dumps(results, indent=2))
