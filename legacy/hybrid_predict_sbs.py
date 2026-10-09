import random
import json
import os
from openai import OpenAI
from transformers import AutoModelForCausalLM
from modelscope import AutoTokenizer
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import torch
import argparse

from evaluate import evaluate
from generate_steps_sbs import generate_steps_sbs
from BCERerank import BCERerank


def hybrid_predict(dataset, use_best_hit=False, retrieve_step_by_step=False, small_model="Qwen2.5-7B", large_model="Qwen2.5-72B"):

    generate_steps_sbs(dataset, large_model)

    if use_best_hit:
        if large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/test_with_72b_generate_steps_sbs.json") as f:
                dev = json.load(f)
        elif large_model == "deepseek-r1":
            with open(f"data/{dataset}/test_with_ds_r1_generate_steps_sbs.json") as f:
                dev = json.load(f)
        elif large_model == "deepseek-v3":
            with open(f"data/{dataset}/test_with_ds_v3_generate_steps_sbs.json") as f:
                dev = json.load(f)
    elif not retrieve_step_by_step:
        # if not os.path.exists(f"data/{dataset}/dev_retrieved_with_72b_generate_steps.json"):
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cuda:0'},
            encode_kwargs={'normalize_embeddings': True},  # set True to compute cosine similarity
            query_instruction="Represent this sentence for searching relevant passages:"
        )
        reranker = BCERerank(
            model="BAAI/bge-reranker-large",
            top_n=10,
            device="cuda:0",
            use_fp16=True
        )
        if large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/test_with_72b_generate_steps.json") as f:
                questions = json.load(f)
        elif large_model == "deepseek-r1":
            with open(f"data/{dataset}/test_with_ds_r1_generate_steps.json") as f:
                questions = json.load(f)
        elif large_model == "deepseek-v3":
            with open(f"data/{dataset}/test_with_ds_v3_generate_steps.json") as f:
                questions = json.load(f)
        retrieved_result = []
        for q in tqdm(questions):
            texts = []
            for ctx in q["ctxs"]:
                for sentence in ctx["sentences"]:
                    texts.append(sentence)
            faiss_vectorstore = FAISS.from_texts(
                texts=texts, embedding=embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )
            faiss_retriever = faiss_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=faiss_retriever
            )
            retrieved_text = ""
            for srcdoc in retriever.get_relevant_documents(q["question"]):
                retrieved_text += srcdoc.page_content + "\n\n"
            q["retrieved_text"] = retrieved_text[:-2]
            retrieved_result.append(q)
        
        if large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/test_retrieved_with_72b_generate_steps.json", 'w', encoding="utf-8") as f:
                json.dump(retrieved_result, f, ensure_ascii=False, indent=4)
        elif large_model == "deepseek-r1":
            with open(f"data/{dataset}/test_with_ds_r1_generate_steps.json","w") as f:
                json.dump(retrieved_result, f, indent=4)
        elif large_model == "deepseek-v3":
            with open(f"data/{dataset}/test_with_ds_v3_generate_steps.json","w") as f:
                json.dump(retrieved_result, f, indent=4)

        dev = retrieved_result

    client = OpenAI(
        api_key="vllm",
        base_url="http://localhost:6010/v1",
    )

    answers = []

    for item in tqdm(dev):
        if use_best_hit:
            text = ""
            ctxs = []
            for ctx in item["ctxs"]:
                ctxs.append(ctx["sentences"])
            for supporting_fact in item["supporting_facts"]:
                if supporting_fact[0] < len(ctxs) and supporting_fact[1] < len(ctxs[supporting_fact[0]]):
                    text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
            text = text[:-1]
        else:
            text = item["retrieved_text"]

        if item["72b_generate_steps"]:
            steps = item["72b_generate_steps"]
        else:
            steps = []

        history = ""
        for step in steps:
            instruction = "You are a helpful assistant. According to the given corpus, follow the given instruction. Output the executing result of the instruction."
            
            response = client.chat.completions.create(
                model=small_model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": "Corpus:\n" + text +  "\n\n" + history + "\n\nInstruction:\n" + step},
                ],
            )
            res = response.choices[0].message.content
            
            history = history + res
        
        response = client.chat.completions.create(
            model=small_model,
            messages=[
                {"role": "system", "content": "According to given information and the given question, output the complete final answer as short as possible. Specially, if the given question is a yes-no question, just output \"yes\" or \"no\"."},
                {"role": "user", "content": history + "\n\nQuestion: " + item["question"]},
            ],
        )
        res = response.choices[0].message.content
        
        answers.append(
            {
                "question": item["question"],
                "answer": item["answers"][0],
                "history": history,
                "pred": res
            }
        )

    if use_best_hit:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/hybrid_pred_7b_72b_sbs_use_best_hit.json","w") as f:
                json.dump(answers, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-r1":
            with open(f"data/{dataset}/hybrid_pred_7b_ds_r1_sbs_use_best_hit.json","w") as f:
                json.dump(answers, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-v3":
            with open(f"data/{dataset}/hybrid_pred_7b_ds_v3_sbs_use_best_hit.json","w") as f:
                json.dump(answers, f, indent=4)
    else:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/hybrid_pred_7b_72b_sbs.json","w") as f:
                json.dump(answers, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-r1":
            with open(f"data/{dataset}/hybrid_pred_7b_ds_r1_sbs.json","w") as f:
                json.dump(answers, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-v3":
            with open(f"data/{dataset}/hybrid_pred_7b_ds_v3_sbs.json","w") as f:
                json.dump(answers, f, indent=4)

    eval_res = evaluate(answers)

    if use_best_hit:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_72b_sbs_use_best_hit.json","w") as f:
                json.dump(eval_res, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-r1":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_ds_r1_sbs_use_best_hit.json","w") as f:
                json.dump(eval_res, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-v3":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_ds_v3_sbs_use_best_hit.json","w") as f:
                json.dump(eval_res, f, indent=4)
    else:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_72b_sbs.json","w") as f:
                json.dump(eval_res, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-r1":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_ds_r1_sbs.json","w") as f:
                json.dump(eval_res, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-v3":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_ds_v3_sbs.json","w") as f:
                json.dump(eval_res, f, indent=4)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--small_model", default="Qwen2.5-7B", type=str)
    parser.add_argument("--large_model", default="Qwen2.5-72B", type=str)
    parser.add_argument("--use_best_hit", action="store_true")
    parser.add_argument("--retrieve_step_by_step", action="store_true")
    args = parser.parse_args()

    hybrid_predict(args.dataset, args.use_best_hit, args.retrieve_step_by_step, args.small_model, args.large_model)