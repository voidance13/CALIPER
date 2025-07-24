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
import sys
sys.path.append('/home/gpt/hgx01_share/lc/Pleias-RAG-Library')
import pleias_rag_interface
# print(pleias_rag_interface.__version__)
from pleias_rag_interface import RAGWithCitations
import nltk
# nltk.download("punkt")
from nltk.tokenize import word_tokenize

from evaluate import evaluate
from BCERerank import BCERerank


def clean_text(text):
    stopwords = {word.strip() for word in open('data_detection/stopwords.txt', encoding='utf-8')}
    segs = jieba.lcut(text)
    segs = list(filter(lambda x: len(x) > 1, segs))
    segs = list(filter(lambda x: x not in stopwords, segs))
    return " ".join(segs)


# 无分类器
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

    # else:
    #     with open(retrieved_path) as file:
    #         dev = json.load(file)


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


def normal_predict(dataset, use_best_hit=False, model_name="Qwen2.5-7B", embed_model="BAAI/bge-base-en-v1.5", rerank_model="BAAI/bge-reranker-large"):

    if use_best_hit:
        with open(f"data/{dataset}/test.json") as f:
            dev = json.load(f)
    else:
        if embed_model=="BAAI/bge-base-en-v1.5" and rerank_model=="BAAI/bge-reranker-large":
            retrieved_path = f"data/{dataset}/test_retrieved.json"
        else:
            retrieved_path = f"data/{dataset}/test_retrieved_embedderft_rerankerft.json"

        dev = retrieve1(retrieved_path, embed_model, rerank_model, dataset)
        # dev = retrieve2(retrieved_path, embed_model, rerank_model, dataset)
        # dev = retrieve3(retrieved_path, dataset)
        

    if model_name == "Qwen2.5-7B":
        client = OpenAI(
            api_key="vllm",
            base_url="http://localhost:6010/v1",
        )
    elif model_name == "Qwen2.5-72B":
        # client = OpenAI(
        #     api_key="vllm",
        #     base_url="http://localhost:6006/v1",
        # )
        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key="",
            base_url="",
        )
        model_name = "qwen2.5-72b-instruct"
    elif model_name == "deepseek-r1" or model_name == "deepseek-v3":
        client = OpenAI(
            api_key="",
            base_url="",
        )
    elif model_name == "Qwen2.5-7B-ft":
        tokenizer = AutoTokenizer.from_pretrained(
            f"fine_tuning/models/{dataset}_Qwen2.5-7b-ft/model",
            # f"fine_tuning/models/hotpotqa_Qwen2.5-7b-ft/model",
            # "/home/gpt/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75/",
            use_fast=False,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"fine_tuning/models/{dataset}_Qwen2.5-7b-ft/model",
            # f"fine_tuning/models/hotpotqa_Qwen2.5-7b-ft/model",
            # "/home/gpt/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75/",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_cache=False,
            # trust_remote_code=True
        ).eval()
    elif model_name == "all-Qwen2.5-7B-ft":
        tokenizer = AutoTokenizer.from_pretrained(
            f"fine_tuning/models/all_Qwen2.5-7b-ft/model",
            use_fast=False,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"fine_tuning/models/all_Qwen2.5-7b-ft/model",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_cache=False,
            # trust_remote_code=True
        ).eval()
    elif model_name == "Qwen2.5-0.5B":
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/gpt/hgx01_share/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
            use_fast=False,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "/home/gpt/hgx01_share/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_cache=False,
            # trust_remote_code=True
        ).eval()
    elif model_name == "Qwen2.5-0.5B-ft":
        tokenizer = AutoTokenizer.from_pretrained(
            f"fine_tuning/models/{dataset}_Qwen2.5-0.5b-ft/model",
            use_fast=False,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"fine_tuning/models/{dataset}_Qwen2.5-0.5b-ft/model",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_cache=False,
            # trust_remote_code=True
        ).eval()
    # elif model_name == "Pleias-350m":
    #     rag = RAGWithCitations(
    #         model_path_or_name="PleIAs/Pleias-RAG-350M"
    #     )
    # elif model_name == "Pleias-1b":
    #     rag = RAGWithCitations(
    #         model_path_or_name="PleIAs/Pleias-RAG-1B"
    #     )

    def predict(messages, model, tokenizer):
        device = "cuda"
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # with torch.no_grad():
        #     outputs = model.generate(**model_inputs, max_length=5000)

        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    answers = []

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
            text = ""
            for sentence in item["retrieved_text"][:5]:
                text += sentence + " "
            text = text[:-1]

        if model_name in ["Qwen2.5-7B-ft", "Qwen2.5-0.5B", "Qwen2.5-0.5B-ft", "all-Qwen2.5-7B-ft"]:
            instruction = "You are a helpful assistant. According to given articles, answer the given question. Think step by step. Output the final answer as short as possible."
            prompt = "Question:\n" + item["question"] + "\n\nCorpus:\n" + text
        # elif model_name in ["Pleias-350m", "Pleias-1b"]:
        #     query = item["question"]
        #     sources = []
        #     ctxs = []
        #     for ctx in item["ctxs"]:
        #         ctxs.append(ctx["sentences"])
        #     for supporting_fact in item["supporting_facts"]:
        #         if supporting_fact[0] < len(ctxs) and supporting_fact[1] < len(ctxs[supporting_fact[0]]):
        #             sources.append({"text": ctxs[supporting_fact[0]][supporting_fact[1]]})
        else:
            instruction = "You are a helpful assistant. According to given articles, answer the given question. Think step by step. Output the executing result of every step. Then output the final answer as short as possible. If the question is a yes-no question, just output \"yes\" or \"no\"."
            prompt = "----------Corpus----------\n" + text + "\n\n----------Question----------\n" + item["question"]
        
        try:
            if model_name in ["Qwen2.5-7B-ft", "Qwen2.5-0.5B", "Qwen2.5-0.5B-ft", "all-Qwen2.5-7B-ft"]:
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt},
                ]
                res = predict(messages, model, tokenizer)
            # elif model_name in ["Pleias-350m", "Pleias-1b"]:
            #     response = rag.generate(query, sources)["processed"]["answer"]
            #     print(response)
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        # {"role": "system", "content": f"You are a helpful assistant. Answer the given question as short as possible according to given articles.\n{example_str}"},
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": prompt},
                    ],
                )

                res = response.choices[0].message.content
        except:
            res = ""

        answers.append(
            {
                "question": item["question"],
                "answer": item["answers"][0],
                "pred": res
            }
        )

    if use_best_hit and model_name == "Qwen2.5-7B":
        with open(f"data/{dataset}/normal_7b_pred_use_best_hit.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit and model_name == "Qwen2.5-7B-ft":
        with open(f"data/{dataset}/normal_7bft_pred_use_best_hit.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit and model_name == "Qwen2.5-0.5B":
        with open(f"data/{dataset}/normal_05b_pred_use_best_hit.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit and model_name == "Qwen2.5-0.5B-ft":
        with open(f"data/{dataset}/normal_05bft_pred_use_best_hit.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit == False and model_name == "Qwen2.5-7B":
        with open(f"data/{dataset}/normal_7b_pred.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit == False and model_name == "Qwen2.5-7B-ft":
        with open(f"data/{dataset}/normal_7bft_pred.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit and model_name == "Qwen2.5-72B":
        with open(f"data/{dataset}/normal_72b_pred_use_best_hit.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit == False and model_name == "Qwen2.5-72B":
        with open(f"data/{dataset}/normal_72b_pred.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit and model_name == "deepseek-r1":
        with open(f"data/{dataset}/normal_ds_r1_pred_use_best_hit.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit == False and model_name == "deepseek-r1":
        with open(f"data/{dataset}/normal_ds_r1_pred.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit and model_name == "deepseek-v3":
        with open(f"data/{dataset}/normal_ds_v3_pred_use_best_hit.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit == False and model_name == "deepseek-v3":
        with open(f"data/{dataset}/normal_ds_v3_pred.json","w") as f:
            json.dump(answers, f, indent=4)
    # elif use_best_hit and model_name == "Pleias-350m":
    #     with open(f"data/{dataset}/normal_pleias_350m_pred_use_best_hit.json","w") as f:
    #         json.dump(answers, f, indent=4)
    # elif use_best_hit and model_name == "Pleias-1b":
    #     with open(f"data/{dataset}/normal_pleias_1b_pred_use_best_hit.json","w") as f:
    #         json.dump(answers, f, indent=4)

    eval_res = evaluate(answers)

    if use_best_hit and model_name == "Qwen2.5-7B":
        with open(f"data/{dataset}/eval_normal_7b_pred_use_best_hit.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit and model_name == "Qwen2.5-7B-ft":
        with open(f"data/{dataset}/eval_normal_7bft_pred_use_best_hit.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit and model_name == "Qwen2.5-0.5B":
        with open(f"data/{dataset}/eval_normal_05b_pred_use_best_hit.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit and model_name == "Qwen2.5-0.5B-ft":
        with open(f"data/{dataset}/eval_normal_05bft_pred_use_best_hit.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit == False and model_name == "Qwen2.5-7B":
        with open(f"data/{dataset}/eval_normal_7b_pred.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit == False and model_name == "Qwen2.5-7B-ft":
        with open(f"data/{dataset}/eval_normal_7bft_pred.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit and model_name == "Qwen2.5-72B":
        with open(f"data/{dataset}/eval_normal_72b_pred_use_best_hit.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit == False and model_name == "Qwen2.5-72B":
        with open(f"data/{dataset}/eval_normal_72b_pred.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit and model_name == "deepseek-r1":
        with open(f"data/{dataset}/eval_normal_ds_r1_pred_use_best_hit.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit == False and model_name == "deepseek-r1":
        with open(f"data/{dataset}/eval_normal_ds_r1_pred.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit and model_name == "deepseek-v3":
        with open(f"data/{dataset}/eval_normal_ds_v3_pred_use_best_hit.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit == False and model_name == "deepseek-v3":
        with open(f"data/{dataset}/eval_normal_ds_v3_pred.json","w") as f:
            json.dump(eval_res, f, indent=4)
    # elif use_best_hit and model_name == "Pleias-350m":
    #     with open(f"data/{dataset}/eval_normal_pleias_350m_pred_use_best_hit.json","w") as f:
    #         json.dump(eval_res, f, indent=4)
    # elif use_best_hit and model_name == "Pleias-1b":
    #     with open(f"data/{dataset}/eval_normal_pleias_1b_pred_use_best_hit.json","w") as f:
    #         json.dump(eval_res, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--model", default="Qwen2.5-7B", type=str)
    parser.add_argument("--use_best_hit", action="store_true")
    parser.add_argument("--embed_model", default="BAAI/bge-base-en-v1.5", type=str)
    parser.add_argument("--rerank_model", default="BAAI/bge-reranker-large", type=str)
    args = parser.parse_args()

    normal_predict(args.dataset, args.use_best_hit, args.model, args.embed_model, args.rerank_model)