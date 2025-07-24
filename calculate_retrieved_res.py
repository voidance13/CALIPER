import json

# with open(f"data/2wikimultihopqa/test_retrieved.json") as file:
#     samples = json.load(file)
with open(f"data/2wikimultihopqa/test_retrieved_embedderft_rerankerft.json") as file:
    samples = json.load(file)
# with open(f"data/hotpotqa/test_retrieved.json") as file:
#     samples = json.load(file)
# with open(f"data/hotpotqa/test_retrieved_embedderft_rerankerft.json") as file:
#     samples = json.load(file)
# with open(f"data/musique/test_retrieved.json") as file:
#     samples = json.load(file)
# with open(f"data/musique/test_retrieved_embedderft_rerankerft.json") as file:
#     samples = json.load(file)

hit3_total = 0
hit5_total = 0
hit10_total = 0
for sample in samples:
    hit3 = 0
    hit5 = 0
    hit10 = 0
    ctxs = []
    for ctx in sample["ctxs"]:
        ctxs.append(ctx["sentences"])
    for supporting_fact in sample["supporting_facts"]:
        if supporting_fact[0] < len(ctxs) and supporting_fact[1] < len(ctxs[supporting_fact[0]]):
            if ctxs[supporting_fact[0]][supporting_fact[1]] in sample["retrieved_text"][:10]:
                hit10 += 1
            if ctxs[supporting_fact[0]][supporting_fact[1]] in sample["retrieved_text"][:5]:
                hit5 += 1
            if ctxs[supporting_fact[0]][supporting_fact[1]] in sample["retrieved_text"][:3]:
                hit3 += 1
    hit3_total += hit3 / len(sample["supporting_facts"])
    hit5_total += hit5 / len(sample["supporting_facts"])
    hit10_total += hit10 / len(sample["supporting_facts"])

print(hit3_total / len(samples))
print(hit5_total / len(samples))
print(hit10_total / len(samples))