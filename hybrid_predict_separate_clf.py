import random
import json
import os
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import jieba
import fasttext
import argparse
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
aliyun_api_key = os.getenv("ALIYUN_API_KEY")

from evaluate import evaluate
from BCERerank import BCERerank


def generate_steps(client, model, question):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Output the solution steps for the given question. Don't use any information that is not given. Don't give any example. Only generate necessary steps succinctly."},
                {"role": "user", "content": question},
            ]
        )
        return response.choices[0].message.content
        # print(res)
    except:
        return ""
    
    
def clean_text(text):
    stopwords = {word.strip() for word in open('model_router/stopwords.txt', encoding='utf-8')}
    segs = jieba.lcut(text)
    segs = list(filter(lambda x: len(x) > 1, segs))
    segs = list(filter(lambda x: x not in stopwords, segs))
    return " ".join(segs)


# w/o model router
def retrieve1(retrieved_path, embed_model, rerank_model, dataset):
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
    with open(f"data/{dataset}/test.json") as file:
        questions = json.load(file)
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


# w/ model router
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

    data_clf = fasttext.load_model(f'model_router/train/{dataset}_train.bin')
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


def hybrid_predict(dataset, use_best_hit=False, small_model="Qwen2.5-7B", large_model="Qwen2.5-72B", embed_model="BAAI/bge-base-en-v1.5", rerank_model="BAAI/bge-reranker-large"):

    if large_model == "Qwen2.5-72B":
        # generate_steps_client = OpenAI(
        #     api_key="vllm",
        #     base_url="http://localhost:6006/v1",
        # )
        generate_steps_client = OpenAI(
            api_key=aliyun_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        large_model_name = "qwen2.5-72b-instruct"
    elif large_model == "llama3-70B":
        generate_steps_client = OpenAI(
            api_key="vllm",
            base_url="http://localhost:6009/v1",
        )
        large_model_name = "llama3.1-70b-instruct"
    elif large_model == "deepseek-r1" or large_model == "deepseek-v3":
        generate_steps_client = OpenAI(
            api_key="",
            base_url="",
        )

    if use_best_hit:
        with open(f"data/{dataset}/test.json") as f:
            dev = json.load(f)
    else:
        retrieved_path = f"data/{dataset}/test_retrieved.json"
        
        dev = retrieve1(retrieved_path, embed_model, rerank_model, dataset)
        # dev = retrieve2(retrieved_path, embed_model, rerank_model, dataset)

    if small_model == "Qwen2.5-7B":
        client = OpenAI(
            api_key="vllm",
            base_url="http://localhost:6010/v1",
        )
        small_model_name = "Qwen2.5-7B"
    elif small_model == "llama3-8B":
        client = OpenAI(
            api_key="vllm",
            base_url="http://localhost:6008/v1",
        )
        small_model_name = "llama3.1-8b-instruct"

    answers = []

    fasttext_model = fasttext.load_model('query_clf/train/train_bi.bin')

    for item in tqdm(dev):
        text = ""
        if use_best_hit:
            ctxs = []
            for ctx in item["ctxs"]:
                ctxs.append(ctx["sentences"])
            for supporting_fact in item["supporting_facts"]:
                if supporting_fact[0] < len(ctxs) and supporting_fact[1] < len(ctxs[supporting_fact[0]]):
                    text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
            text = text[:-1]
        else:
            for sentence in item["retrieved_text"][:5]:
                text += sentence + " "
            text = text[:-1]

        question = item["question"]
        pred = fasttext_model.predict(question)[0]
        
        if "__label__easy" in pred:
            instruction1 = "You are a helpful assistant. According to given articles, answer the given question. Think step by step. Output the executing result of every step."
            prompt1 = "Question:\n" + item["question"] + "\n\nCorpus:\n" + text
            instruction2 = "According to the given question and the reasoning steps, output the complete final answer as short as possible. Specially, if the given question is a yes-no question, just output \"yes\" or \"no\"."
            
            response = client.chat.completions.create(
                model=small_model_name,
                messages=[
                    {"role": "system", "content": instruction1},
                    {"role": "user", "content": prompt1},
                ],
            )
            res1 = response.choices[0].message.content

            response = client.chat.completions.create(
                model=small_model_name,
                messages=[
                    {"role": "system", "content": instruction2},
                    {"role": "user", "content": "----------Question----------\n" + item["question"] + "\n\n----------Steps----------\n" + res1},
                ],
            )
            res = response.choices[0].message.content
                
            answers.append(
                {
                    "question": item["question"],
                    "answer": item["answers"][0],
                    "res1": res1,
                    "pred": res
                }
            )
        else:
            steps = generate_steps(generate_steps_client, large_model_name, question)

            instruction1 = "You are a helpful assistant. According to the given steps and corpus, answer the given question. Output the executing result of every step in simple sentences."
            prompt = "----------Question----------\n" + item["question"] + "----------Corpus----------\n" + text + "\n\n----------Steps----------\n" + steps + "\n\n----------Question----------\n" + item["question"]
            instruction2 = "According to the given question and reasoning steps, output the final answer as short as possible."

            response = client.chat.completions.create(
                model=small_model_name,
                messages=[
                    {"role": "system", "content": instruction1},
                    {"role": "user", "content": prompt},
                ],
            )
            res1 = response.choices[0].message.content

            response = client.chat.completions.create(
                model=small_model_name,
                messages=[
                    {"role": "system", "content": instruction2},
                    {"role": "user", "content": "----------Steps----------\n" + res1 + "\n\n----------Question----------\n" + item["question"]},
                ],
            )
            res = response.choices[0].message.content
        
            answers.append(
                {
                    "question": item["question"],
                    "answer": item["answers"][0],
                    "steps": steps,
                    "res1": res1,
                    "pred": res
                }
            )

    if use_best_hit:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/hybrid_pred_7b_72b_sep_clf_use_best_hit.json","w") as f:
                json.dump(answers, f, indent=4)
        if small_model == "llama3-8B" and large_model == "llama3-70B":
            with open(f"data/{dataset}/hybrid_pred_llama3_8b_70b_sep_clf_use_best_hit.json","w") as f:
                json.dump(answers, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-r1":
            with open(f"data/{dataset}/hybrid_pred_7b_ds_r1_sep_clf_use_best_hit.json","w") as f:
                json.dump(answers, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-v3":
            with open(f"data/{dataset}/hybrid_pred_7b_ds_v3_sep_clf_use_best_hit.json","w") as f:
                json.dump(answers, f, indent=4)
    else:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/hybrid_pred_7b_72b_sep_clf.json","w") as f:
                json.dump(answers, f, indent=4)
        if small_model == "llama3-8B" and large_model == "llama3-70B":
            with open(f"data/{dataset}/hybrid_pred_llama3_8b_70b_sep_clf.json","w") as f:
                json.dump(answers, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-r1":
            with open(f"data/{dataset}/hybrid_pred_7b_ds_r1_sep_clf.json","w") as f:
                json.dump(answers, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-v3":
            with open(f"data/{dataset}/hybrid_pred_7b_ds_v3_sep_clf.json","w") as f:
                json.dump(answers, f, indent=4)

    eval_res = evaluate(answers)

    if use_best_hit:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_72b_sep_clf_use_best_hit.json","w") as f:
                json.dump(eval_res, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-r1":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_ds_r1_sep_clf_use_best_hit.json","w") as f:
                json.dump(eval_res, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-v3":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_ds_v3_sep_clf_use_best_hit.json","w") as f:
                json.dump(eval_res, f, indent=4)
    else:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_72b_sep_clf.json","w") as f:
                json.dump(eval_res, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-r1":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_ds_r1_sep_clf.json","w") as f:
                json.dump(eval_res, f, indent=4)
        elif small_model == "Qwen2.5-7B" and large_model == "deepseek-v3":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_ds_v3_sep_clf.json","w") as f:
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