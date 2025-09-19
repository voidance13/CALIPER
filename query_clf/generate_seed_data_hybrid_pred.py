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


def generate_steps(dataset, large_model):

    with open(f"data/{dataset}/test.json") as f:
        dev = json.load(f)

    if large_model == "Qwen2.5-72B":
        client = OpenAI(
            api_key="vllm",
            base_url="http://localhost:6006/v1",
        )
    elif large_model == "deepseek-r1" or large_model == "deepseek-v3":
        client = OpenAI(
            api_key="",
            base_url="",
        )

    result = []

    for sample in tqdm(dev[:2000]):
        question = sample["question"]
        
        try:
            response = client.chat.completions.create(
                model=large_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Just output steps to answer the given question. Don't use any information that is not given. Don't give any example."},
                    {"role": "user", "content": question},
                ]
            )
            res = response.choices[0].message.content
            # print(res)
        except:
            res = ""

        sample["72b_generate_steps"] = res
        result.append(sample)
        
    return result


def hybrid_predict(dataset, use_best_hit=False, retrieve_step_by_step=False, small_model="Qwen2.5-7B", large_model="Qwen2.5-72B"):

    dev = generate_steps(dataset, large_model)

    client = OpenAI(
        api_key="vllm",
        base_url="http://localhost:6010/v1",
    )

    answers = []

    for item in tqdm(dev[:2000]):
        if use_best_hit:
            text = ""
            ctxs = []
            for ctx in item["ctxs"]:
                ctxs.append(ctx["sentences"])
            for supporting_fact in item["supporting_facts"]:
                text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
            text = text[:-1]
        else:
            text = item["retrieved_text"]

        if item["72b_generate_steps"]:
            steps = item["72b_generate_steps"]
        else:
            steps = ""
        
        instruction = "You are a helpful assistant. According to given steps and articles, answer the given question. Output the executing result of every step. Then output the final answer as short as possible."
        prompt = "Question:\n" + item["question"] + "\n\nSteps:\n" + steps + "\n\nCorpus:\n" + text
        # instruction = "You are a helpful assistant. According to given steps and articles, answer the given question. Output the executing result of every step. Then output the final answer as short as possible. If the question is a yes-no question, just output \"yes\" or \"no\"."
        # prompt = "Question:\n" + item["question"] + "\n\nSteps:\n" + steps + "\n\nCorpus:\n" + text

        response = client.chat.completions.create(
            model=small_model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
        )

        res = response.choices[0].message.content
        # print(res)
        
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

    answers1 = hybrid_predict("musique", True, False, "Qwen2.5-7B", "Qwen2.5-72B")
    easy1 = []
    for sample in answers1:
        if sample["pred"]:
            pred = parse_answer(sample["pred"])

            hit = eval_hit(pred, sample["answer"])

            if hit == 0:
                # print(pred)
                easy1.append(sample["id"])

    with open("query_clf/musique_test2000_72b_7b_wrong1.json","w") as f:
        json.dump(easy1, f, indent=4)

    
    answers1 = hybrid_predict("musique", True, False, "Qwen2.5-7B", "Qwen2.5-72B")
    easy1 = []
    for sample in answers1:
        if sample["pred"]:
            pred = parse_answer(sample["pred"])

            hit = eval_hit(pred, sample["answer"])

            if hit == 0:
                # print(pred)
                easy1.append(sample["id"])

    with open("query_clf/musique_test2000_72b_7b_wrong2.json","w") as f:
        json.dump(easy1, f, indent=4)


    answers1 = hybrid_predict("musique", True, False, "Qwen2.5-7B", "Qwen2.5-72B")
    easy1 = []
    for sample in answers1:
        if sample["pred"]:
            pred = parse_answer(sample["pred"])

            hit = eval_hit(pred, sample["answer"])

            if hit == 0:
                # print(pred)
                easy1.append(sample["id"])

    with open("query_clf/musique_test2000_72b_7b_wrong3.json","w") as f:
        json.dump(easy1, f, indent=4)
