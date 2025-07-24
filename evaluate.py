import regex
import string
from collections import Counter
import json
import re
from sentence_transformers import SentenceTransformer, util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model = SentenceTransformer('paraphrase-mpnet-base-v2')


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    # if normalized_prediction in ['yes', 'no'] and normalized_prediction != normalized_ground_truth:
    #     return ZERO_METRIC
    # if normalized_ground_truth in ['yes', 'no'] and normalized_prediction != normalized_ground_truth:
    #     return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def eval_hit(prediction, answer):
    if normalize_answer(answer) in normalize_answer(prediction):
        return 1
    return 0


def parse_answer(response):
    try:
        return response[re.search("Answer:\n", response).end():]
    except:
        try: 
            return response[re.search("answer:\n", response).end():]
        except:
            try:
                return response[re.search("Answer: ", response).end():]
            except:
                try:
                    return response[re.search("answer: ", response).end():]
                except:
                    try:
                        return response[re.search("Answer:", response).end():]
                    except:
                        try:
                            return response[re.search("answer:", response).end():]
                        except:
                            line = response.split("\n")[-1]
                            try:
                                return line[re.search(": ", line).end():]
                            except:
                                return line


def evaluate(samples):

    SBERT_COS = 0
    HIT = 0
    PRECISION = 0
    RECALL = 0
    F1 = 0

    result = []

    for sample in samples:
        if sample["pred"]:
            pred = parse_answer(sample["pred"])
            embedding1 = model.encode(pred, convert_to_tensor=True)
            embedding2 = model.encode(sample["answer"], convert_to_tensor=True)
            cosine_score = util.pytorch_cos_sim(embedding1, embedding2)

            f1, precision, recall = f1_score(pred, sample["answer"])
            hit = eval_hit(pred, sample["answer"])
            # if hit == 0:
            #     print(sample["answer"])
            SBERT_COS += cosine_score
            HIT += hit
            PRECISION += precision
            RECALL += recall
            F1 += f1

            sample["SBERT similarity"] = cosine_score.item()
        result.append(sample)

    print("SBERT cosine score", SBERT_COS / len(samples))
    print("hit", HIT / len(samples))
    print("precision", PRECISION / len(samples))
    print("recall", RECALL / len(samples))
    print("f1", F1 / len(samples))

    return result


# with open("data/hotpotqa/normal_ds_r1_pred_use_best_hit.json") as file:
#     answers = json.load(file)

# eval_res = evaluate(answers)

# with open("data/hotpotqa/eval_normal_ds_r1_pred_use_best_hit.json","w") as f:
#     json.dump(eval_res, f, indent=4)