import argparse
import json

import jsonlines


def rerank_ft_preprocess(dataset):
    with open(f"data/{dataset}/train.json") as f:
        train = json.load(f)

    max_length = 330
    final_train_data = []
    for sample in train:
        pos = []
        neg = []
        for ctx in sample["ctxs"]:
            title = ctx["title"]
            for j, sentence in enumerate(ctx["sentences"]):
                if len(sentence) > max_length:
                    sentence = sentence[:max_length]
                if [title, j] in sample["supporting_facts"]:
                    pos.append(sentence)
                else:
                    neg.append(sentence)

        query = sample["question"]
        if len(query) > max_length:
            query = query[:max_length]

        if len(neg) != 0:
            final_train_data.append(
                {
                    "query": query,
                    "pos": pos,
                    "neg": neg,
                }
            )

    with jsonlines.open(f"retrieve_fine_tuning/{dataset}_bge-reranker-ft.jsonl", "w") as f:
        f.write(final_train_data)
    print(dataset, "reranker samples:", len(final_train_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    args = parser.parse_args()
    rerank_ft_preprocess(args.dataset)
