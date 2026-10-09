import glob
import json


patterns = [
    "data/hotpotqa/hotpotqa_caliper_planA_top5_reflA_full.json*",
    "data/hotpotqa/hotpotqa_v0_opt_reflect.json.s*",
    "data/hotpotqa/hotpotqa_v0_ft_steps_reflect.json.s*",
    "data/musique/musique_v0_2stream_steps_reflect.json*",
    "data/musique/musique_v0_opt_reflect.json*",
    "data/2wikimultihopqa/2wikimultihopqa_v0_ft_steps_reflect.json.s*",
    "data/2wikimultihopqa/2wikimultihopqa_v0_opt_reflect.json.s*",
]

for pattern in patterns:
    files = [name for name in sorted(glob.glob(pattern)) if not name.endswith(".tmp")]
    sets = []
    print("\n", pattern)
    for name in files:
        rows = json.load(open(name))
        questions = {row.get("question", "") for row in rows}
        sets.append(questions)
        print(name, "n", len(rows), "unique", len(questions), "empty", sum(not row.get("pred", "").strip() for row in rows), "keys", list(rows[0]) if rows else [])
    print("union", len(set().union(*sets)) if sets else 0)
    for i in range(len(sets)):
        print("overlap", i, [len(sets[i] & sets[j]) for j in range(i)])
