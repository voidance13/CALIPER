import random
import json
import sys
import os
import re
import regex
import string
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import argparse

sys.path.append(".")
from BCERerank import BCERerank


def normal_predict(dataset, use_best_hit=True, model="Qwen2.5-7B"):

    with open(f"data/{dataset}/test.json") as f:
        dev = json.load(f)

    if model == "Qwen2.5-7B":
        client = OpenAI(
            api_key="vllm",
            base_url="http://localhost:6010/v1",
        )
    elif model == "Qwen2.5-72B":
        client = OpenAI(
            api_key="vllm",
            base_url="http://localhost:6006/v1",
        )
    elif model == "deepseek-r1" or model == "deepseek-v3":
        client = OpenAI(
            api_key="",
            base_url="",
        )

    answers = []

    # example_str = ""
    # for i, example in enumerate(dev[:3]):
    #     question = example["question"]
    #     corpus = ""
    #     for ctx in example["ctxs"]:
    #         corpus += ctx["text"] + "\n\n"
    #     answer = example["answers"][0]
    #     example_str += f"\n\nExample{str(i+1)}:\nInput:\nQuestion:\n{question}\n\nCorpus:\n{corpus}\n\nOutput:\n{answer}"

    for item in tqdm(dev[:2000]):
        text = ""
        if use_best_hit:
            if dataset == "news_articles":
                for evidence in item["evidence_list"]:
                    text += evidence["fact"] + "\n"
            else:
                ctxs = []
                for ctx in item["ctxs"]:
                    ctxs.append(ctx["sentences"])
                for supporting_fact in item["supporting_facts"]:
                    text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
            text = text[:-1]
        else:
            text = item["retrieved_text"]

        if dataset == "news_articles":
            # instruction = "You are a helpful assistant. According to given steps and articles, answer the given question. Output the executing result of every step. Then output the final answer as short as possible. If given corpus is not sufficient enough to answer the question, output \"Insufficient information.\". If the question is a yes-no question, just output \"yes\", \"no\" or \"Insufficient information.\"."
            # prompt = "Question:\n" + item["query"] + "\n\nCorpus:\n" + text
            instruction = "You are a helpful assistant. According to given articles, answer the given question."
            prompt = "Question:\n" + item["query"] + "\n\nCorpus:\n" + text + "\n\nThink step by step. Output the executing result of every step. Then output the final answer as short as possible. If given corpus is not sufficient enough to answer the question, output \"Insufficient information.\". If the question is a yes-no question, just output \"yes\", \"no\" or \"Insufficient information.\"."
        else:
            instruction = "You are a helpful assistant. According to given steps and articles, answer the given question. Think step by step. Output the executing result of every step. Then output the final answer as short as possible. If the question is a yes-no question, just output \"yes\" or \"no\"."
            prompt = "Question:\n" + item["question"] + "\n\nCorpus:\n" + text
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    # {"role": "system", "content": f"You are a helpful assistant. Answer the given question as short as possible according to given articles.\n{example_str}"},
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt},
                ],
            )

            res = response.choices[0].message.content
        except:
            res = ""

        if dataset == "news_articles":
            answers.append(
                {
                    "id": item["id"],
                    "question": item["query"],
                    "answer": item["answer"],
                    "pred": res
                }
            )
        else:
            answers.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answers"][0],
                    "pred": res
                }
            )

    return answers


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


def eval_hit(prediction, answer):
    if normalize_answer(answer) in normalize_answer(prediction):
        return 1
    return 0


if __name__ == "__main__":

    answers1 = normal_predict("musique", True, "Qwen2.5-7B")
    easy1 = []
    for sample in answers1:
        if sample["pred"]:
            pred = parse_answer(sample["pred"])

            hit = eval_hit(pred, sample["answer"])

            if hit == 1:
                # print(pred)
                easy1.append(sample["id"])

    with open("query_clf/musique_test2000_7b_right1.json","w") as f:
        json.dump(easy1, f, indent=4)

    
    answers1 = normal_predict("musique", True, "Qwen2.5-7B")
    easy1 = []
    for sample in answers1:
        if sample["pred"]:
            pred = parse_answer(sample["pred"])

            hit = eval_hit(pred, sample["answer"])

            if hit == 1:
                # print(pred)
                easy1.append(sample["id"])

    with open("query_clf/musique_test2000_7b_right2.json","w") as f:
        json.dump(easy1, f, indent=4)


    answers1 = normal_predict("musique", True, "Qwen2.5-7B")
    easy1 = []
    for sample in answers1:
        if sample["pred"]:
            pred = parse_answer(sample["pred"])

            hit = eval_hit(pred, sample["answer"])

            if hit == 1:
                # print(pred)
                easy1.append(sample["id"])

    with open("query_clf/musique_test2000_7b_right3.json","w") as f:
        json.dump(easy1, f, indent=4)
