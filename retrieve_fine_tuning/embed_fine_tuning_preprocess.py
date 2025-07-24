import jsonlines
import argparse
import torch
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 7"


def rerank_ft_preprocess(dataset):

    with open(f"data/{dataset}/train.json") as f:
        train = json.load(f)

    # max_length = 320
    final_train_data = []
    for sample in train:
        pos = []
        neg = []
        ctxs = []
        for ctx in sample["ctxs"]:
            ctxs.append(ctx["sentences"])
        for i in range(len(ctxs)):
            for j in range(len(ctxs[i])):
                # if len(ctxs[i][j]) > max_length:
                #     ctxs[i][j] = ctxs[i][j][:max_length]
                if [i, j] in sample["supporting_facts"]:
                    pos.append(ctxs[i][j])
                else:
                    neg.append(ctxs[i][j])

        query = sample["question"]
        # if len(query) > max_length:
        #     query = query[:max_length]
        
        if len(neg) != 0:
            final_train_data.append(
                {
                    "query": query,
                    "pos": pos,
                    "neg": neg,
                }
            )

    with jsonlines.open(f"retrieve_fine_tuning/{dataset}_bge-embedder-ft.jsonl", "w") as f:
        f.write(final_train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    args = parser.parse_args()

    rerank_ft_preprocess(args.dataset)
    