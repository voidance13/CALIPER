import random
import json
import os
from openai import OpenAI
from transformers import AutoModelForCausalLM
from modelscope import AutoTokenizer
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import jieba
import fasttext
import torch
import argparse
from nltk.tokenize import word_tokenize

from evaluate import evaluate
from generate_steps import generate_steps
from BCERerank import BCERerank


def clean_text(text):
    stopwords = {word.strip() for word in open('data_detection/stopwords.txt', encoding='utf-8')}
    segs = jieba.lcut(text)
    segs = list(filter(lambda x: len(x) > 1, segs))
    segs = list(filter(lambda x: x not in stopwords, segs))
    return " ".join(segs)


# 无分类器
def retrieve1(large_model, retrieved_path, embed_model, rerank_model, dataset):
    # if not os.path.exists(retrieved_path):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embed_model,
        model_kwargs={'device': 'cuda:0'},
        encode_kwargs={'normalize_embeddings': True},  # set True to compute cosine similarity
        query_instruction="Represent this sentence for searching relevant passages:"
    )
    reranker = BCERerank(
        model=rerank_model,
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

        retrieved_text = []
        for srcdoc in retriever.get_relevant_documents(q["question"]):
            retrieved_text.append(srcdoc.page_content)
        q["retrieved_text"] = retrieved_text
        retrieved_result.append(q)

    with open(retrieved_path, 'w', encoding="utf-8") as f:
        json.dump(retrieved_result, f, ensure_ascii=False, indent=4)

    return retrieved_result


# 有分类器
def retrieve2(retrieved_path, embed_model, rerank_model, dataset):
    ori_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cuda:0'},
        encode_kwargs={'normalize_embeddings': True},  # set True to compute cosine similarity
        query_instruction="Represent this sentence for searching relevant passages:"
    )
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embed_model,
        model_kwargs={'device': 'cuda:0'},
        encode_kwargs={'normalize_embeddings': True},  # set True to compute cosine similarity
        query_instruction="Represent this sentence for searching relevant passages:"
    )
    ori_reranker = BCERerank(
        model="BAAI/bge-reranker-large",
        top_n=10,
        device="cuda:0",
        use_fp16=True
    )
    reranker = BCERerank(
        model=rerank_model,
        top_n=10,
        device="cuda:0",
        use_fp16=True
    )
    with open(f"data/{dataset}/test.json") as file:
        questions = json.load(file)
    retrieved_result = []

    # 数据分类器判断query类型，决定使用的模型
    data_clf = fasttext.load_model(f'data_detection/train/{dataset}_train.bin')
    for q in tqdm(questions):
        if "__label__0" in data_clf.predict(clean_text(q["question"]))[0]:
            e = ori_embeddings
            r = ori_reranker
        else:
            e = embeddings
            r = reranker

        texts = []
        for ctx in q["ctxs"]:
            for sentence in ctx["sentences"]:
                texts.append(sentence)

        bm25_retriever = BM25Retriever.from_texts(texts, preprocess_func=word_tokenize)
        bm25_retriever.k = 10

        faiss_vectorstore = FAISS.from_texts(
            texts=texts, embedding=e, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )
        faiss_retriever = faiss_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        
        retriever = ContextualCompressionRetriever(
            base_compressor=r, base_retriever=ensemble_retriever
        )
        retrieved_text = []
        for srcdoc in retriever.get_relevant_documents(q["question"]):
            retrieved_text.append(srcdoc.page_content)
        q["retrieved_text"] = retrieved_text
        retrieved_result.append(q)

    with open(retrieved_path, 'w', encoding="utf-8") as f:
        json.dump(retrieved_result, f, ensure_ascii=False, indent=4)

    return retrieved_result


# 分词检索
def retrieve3(retrieved_path, dataset):
    with open(f"data/{dataset}/test.json") as file:
        questions = json.load(file)
    retrieved_result = []

    for q in tqdm(questions):

        texts = []
        for ctx in q["ctxs"]:
            ctx.pop('ranked_prompt_indices') 
            for sentence in ctx["sentences"]:
                texts.append(sentence)

        bm25_retriever = BM25Retriever.from_texts(texts, preprocess_func=word_tokenize)
        bm25_retriever.k = 10

        retrieved_text = []
        for srcdoc in bm25_retriever.get_relevant_documents(q["question"]):
            retrieved_text.append(srcdoc.page_content)
        q["retrieved_text"] = retrieved_text
        retrieved_result.append(q)

    with open(retrieved_path, 'w', encoding="utf-8") as f:
        json.dump(retrieved_result, f, ensure_ascii=False, indent=4)

    return retrieved_result


def hybrid_predict(dataset, use_best_hit=False, small_model="Qwen2.5-7B", large_model="Qwen2.5-72B", embed_model="BAAI/bge-base-en-v1.5", rerank_model="BAAI/bge-reranker-large"):

    # generate_steps(dataset, large_model)

    if use_best_hit:
        if large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/test_with_72b_generate_steps.json") as f:
                dev = json.load(f)
    else:
        if large_model == "Qwen2.5-72B":
            retrieved_path = f"data/{dataset}/test_with_72b_generate_steps_retrieved.json"
        elif large_model == "deepseek-r1":
            retrieved_path = f"data/{dataset}/test_with_ds_r1_generate_steps_retrieved.json"
        elif large_model == "deepseek-v3":
            retrieved_path = f"data/{dataset}/test_with_ds_v3_generate_steps_retrieved.json"

        dev = retrieve1(large_model, retrieved_path, embed_model, rerank_model, dataset)

    if small_model == "Qwen2.5-7B":
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
            text = ""
            for sentence in item["retrieved_text"][:5]:
                text += sentence + " "
            text = text[:-1]

        if item["72b_generate_steps"]:
            steps = item["72b_generate_steps"]
        else:
            steps = ""

        instruction1 = "You are a helpful assistant. According to the given steps and corpus, answer the given question. Output the executing result of every step in simple sentences."
        prompt = "----------Question----------\n" + item["question"] + "----------Corpus----------\n" + text + "\n\n----------Steps----------\n" + steps + "\n\n----------Question----------\n" + item["question"]
        instruction2 = "According to the given question and reasoning steps, output the final answer as short as possible."

        response = client.chat.completions.create(
            model=small_model,
            messages=[
                {"role": "system", "content": instruction1},
                {"role": "user", "content": prompt},
            ],
        )
        res1 = response.choices[0].message.content

        response = client.chat.completions.create(
            model=small_model,
            messages=[
                {"role": "system", "content": instruction2},
                {"role": "user", "content": "----------Steps----------\n" + res1 + "\n\n----------Question----------\n" + item["question"]},
            ],
        )
        res = response.choices[0].message.content
        # print(res)

        answers.append(
            {
                "question": item["question"],
                "answer": item["answers"][0],
                "res1": res1,
                "pred": res
            }
        )

    if use_best_hit:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/hybrid_pred_7b_72b_sep_use_best_hit.json","w") as f:
                json.dump(answers, f, indent=4)
    else:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/hybrid_pred_7b_72b_sep.json","w") as f:
                json.dump(answers, f, indent=4)

    eval_res = evaluate(answers)

    if use_best_hit:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_72b_sep_use_best_hit.json","w") as f:
                json.dump(eval_res, f, indent=4)
    else:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_72b_sep.json","w") as f:
                json.dump(eval_res, f, indent=4)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--small_model", default="Qwen2.5-7B", type=str)
    parser.add_argument("--large_model", default="Qwen2.5-72B", type=str)
    parser.add_argument("--use_best_hit", action="store_true")
    parser.add_argument("--embed_model", default="BAAI/bge-base-en-v1.5", type=str)
    parser.add_argument("--rerank_model", default="BAAI/bge-reranker-large", type=str)
    args = parser.parse_args()

    hybrid_predict(args.dataset, args.use_best_hit, args.small_model, args.large_model, args.embed_model, args.rerank_model)