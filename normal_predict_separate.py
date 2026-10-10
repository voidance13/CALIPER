import random
import json
import os
from openai import OpenAI
from transformers import AutoModelForCausalLM
try:
    from modelscope import AutoTokenizer
except ImportError:
    AutoTokenizer = None
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
from concurrent.futures import ThreadPoolExecutor
# import sys
# sys.path.append('/home/gpt/hgx01_share/lc/Pleias-RAG-Library')
# import pleias_rag_interface
# print(pleias_rag_interface.__version__)
# # from pleias_rag_interface import RAGWithCitations
import nltk
# nltk.download("punkt")
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
aliyun_api_key = os.getenv("ALIYUN_API_KEY")
siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")

from evaluate import evaluate
from BCERerank import BCERerank


def clean_text(text):
    stopwords = {word.strip() for word in open('model_router/stopwords.txt', encoding='utf-8')}
    segs = jieba.lcut(text)
    segs = list(filter(lambda x: len(x) > 1, segs))
    segs = list(filter(lambda x: x not in stopwords, segs))
    return " ".join(segs)


def load_cached_retrieval(retrieved_path):
    if os.path.exists(retrieved_path):
        with open(retrieved_path, encoding="utf-8") as f:
            print(f"Loading cached retrieval from {retrieved_path}")
            return json.load(f)
    return None


# w/o model router
def retrieve1(retrieved_path, embed_model, rerank_model, dataset, device="cuda:0"):
    cached = load_cached_retrieval(retrieved_path)
    if cached is not None:
        return cached
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embed_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True},  # set True to compute cosine similarity
        query_instruction="Represent this sentence for searching relevant passages:"
    )
    reranker = BCERerank(
        model=rerank_model,
        top_n=10,
        device=device,
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


def retrieve2(retrieved_path, embed_model, rerank_model, dataset, device="cuda:0"):
    cached = load_cached_retrieval(retrieved_path)
    if cached is not None:
        return cached
    ori_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True},  # set True to compute cosine similarity
        query_instruction="Represent this sentence for searching relevant passages:"
    )
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embed_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True},  # set True to compute cosine similarity
        query_instruction="Represent this sentence for searching relevant passages:"
    )
    ori_reranker = BCERerank(
        model="BAAI/bge-reranker-large",
        top_n=10,
        device=device,
        use_fp16=True
    )
    reranker = BCERerank(
        model=rerank_model,
        top_n=10,
        device=device,
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


# re-write query
def retrieve3(retrieved_path, embed_model, rerank_model, dataset):
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
        
        client = OpenAI(
            api_key="vllm",
            base_url="http://localhost:6009/v1",
        )
        model_name = "llama3.1-70b-instruct"
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Rewrite the given question in another way without changing the meaning. Only output the result of rewriting without explaining."},
                {"role": "user", "content": q["question"]},
            ],
        )
        rewrite_query = response.choices[0].message.content
        print(q["question"])
        print(rewrite_query)
        
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

        # t = rewrite_query
        t = q["question"] + " " + rewrite_query
        retrieved_text = []
        for srcdoc in retriever.get_relevant_documents(t[:200]):
            retrieved_text.append(srcdoc.page_content)
        q["retrieved_text"] = retrieved_text
        retrieved_result.append(q)

    with open(retrieved_path, 'w', encoding="utf-8") as f:
        json.dump(retrieved_result, f, ensure_ascii=False, indent=4)

    return retrieved_result


# llm Purification
def retrieve4(retrieved_path, embed_model, rerank_model, dataset):
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
        
    purification_instruction = "Given a question and a list of sentences, judge which sentences are helpful for answering the question and output these sentences. Only output the helpful sentences. Don't output any other unneccessary words."
    # INSTRUCTION = "给定一个问题和若干条不同的规则，规则包括规则编号和文本信息。规则的输入格式为：“规则编号：规则文本”。请根据给定的问题和规则，返回对回答问题的帮助最大的一条规则，输出他的规则编号。输出格式为“规则编号：x”，不要输出编号列表以外的文字。输出示例：“规则编号：463”。"
    
    client = OpenAI(
        api_key="vllm",
        base_url="http://localhost:6008/v1",
    )
    model_name = "llama3.1-8b-instruct"

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
        
        text = ""
        for t in retrieved_text:
            text += t + " "    
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": purification_instruction},
                {"role": "user", "content": "-----------Question-----------\n" + q["question"] + "\n\n-----------Sentences-----------\n" + text[:-1]},
            ],
        )
        res = response.choices[0].message.content
        print(res)
        retrieved_text = [res]
        
        q["retrieved_text"] = retrieved_text
        
        retrieved_result.append(q)

    with open(retrieved_path, 'w', encoding="utf-8") as f:
        json.dump(retrieved_result, f, ensure_ascii=False, indent=4)

    return retrieved_result


def normal_predict(dataset, use_best_hit=False, model="Qwen3.5-9B", embed_model="BAAI/bge-base-en-v1.5", rerank_model="BAAI/bge-reranker-large", max_workers=8, device="cuda:0", tag="", limit=0, no_thinking=False):
    tag_suffix = f"_{tag}" if tag else ""

    if use_best_hit:
        with open(f"data/{dataset}/test.json") as f:
            dev = json.load(f)
    else:
        if (embed_model.endswith("bge-base-en-v1.5") and rerank_model.endswith("bge-reranker-large")):
            retrieved_path = f"data/{dataset}/test_retrieved.json"
        else:
            retrieved_path = f"data/{dataset}/test_retrieved_embedderft_rerankerft.json"

        # Base retriever: bge-base-en-v1.5 + bge-reranker-large (Table 2 setting).
        dev = retrieve1(retrieved_path, embed_model, rerank_model, dataset, device=device)
        # dev = retrieve2(retrieved_path, embed_model, rerank_model, dataset)
        # dev = retrieve3(retrieved_path, embed_model, rerank_model, dataset)
        # dev = retrieve4(retrieved_path, embed_model, rerank_model, dataset)
        
    if model == "Qwen2.5-0.5B":
        max_workers = 1

    if limit > 0:
        dev = dev[:limit]

    if model in ("Qwen3.5-9B", "qwen3.5-9b"):
        client = OpenAI(
            api_key=os.getenv("CALIPER_LOCAL_API_KEY", "vllm"),
            base_url=os.getenv("CALIPER_LOCAL_BASE_URL") or "http://localhost:6010/v1",
        )
        model_name = os.getenv("CALIPER_LOCAL_MODEL_NAME", "Qwen3.5-9B")
    elif model == "Qwen2.5-7B":
        client = OpenAI(
            api_key="vllm",
            base_url="http://localhost:6010/v1",
        )
        model_name = "Qwen2.5-7B"
    elif model == "Qwen2.5-72B":
        client = OpenAI(
            api_key=aliyun_api_key,
            base_url=os.getenv("CALIPER_CLOUD_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        model_name = "qwen3.7-plus"
    elif model == "deepseek-r1":
        client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com",
        )
        model_name = "deepseek-reasoner"
    elif model == "deepseek-v3":
        client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com",
        )
        model_name = "deepseek-chat"
    elif model == "Qwen2.5-0.5B":
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
    elif model == "llama3-8B":
        client = OpenAI(
            api_key="vllm",
            base_url="http://localhost:6008/v1",
        )
        model_name = "llama3.1-8b-instruct"
    elif model == "llama3-70B":
        client = OpenAI(
            api_key="vllm",
            base_url="http://localhost:6009/v1",
        )
        model_name = "llama3.1-70b-instruct"

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

        return response

    def process_item(idx, item):
        text = ""
        if use_best_hit:
            # Table 3 gold context: keep only supporting facts.
            ctxs = [ctx["sentences"] for ctx in item["ctxs"]]
            title_to_idx = {ctx["title"]: i for i, ctx in enumerate(item["ctxs"])}
            for title, sent_idx in item["supporting_facts"]:
                ci = title_to_idx.get(title)
                if ci is not None and sent_idx < len(ctxs[ci]):
                    text += ctxs[ci][sent_idx] + "\n"
            text = text[:-1]
        else:
            for sentence in item["retrieved_text"][:5]:
                text += sentence + " "
            text = text[:-1]

        # if model_name in ["Qwen2.5-7B-ft", "Qwen2.5-0.5B", "Qwen2.5-0.5B-ft", "all-Qwen2.5-7B-ft"]:
        #     instruction = "You are a helpful assistant. According to given articles, answer the given question. Think step by step. Output the final answer as short as possible."
        #     prompt = "Question:\n" + item["question"] + "\n\nCorpus:\n" + text
        # else:
        instruction1 = "You are a helpful assistant. According to given articles, answer the given question. Think step by step. Output the executing result of every step."
        prompt1 = "Question:\n" + item["question"] + "\n\nCorpus:\n" + text
        instruction2 = "According to the given question and the reasoning steps, output the complete final answer as short as possible. Specially, if the given question is a yes-no question, just output \"yes\" or \"no\"."
        cloud_kwargs = {}
        if no_thinking and model == "Qwen2.5-72B":
            cloud_kwargs["extra_body"] = {"enable_thinking": False}
        
        res1 = ""
        res = ""
        for attempt in range(2):
            try:
                # if model_name in ["Qwen2.5-7B-ft", "Qwen2.5-0.5B", "Qwen2.5-0.5B-ft", "all-Qwen2.5-7B-ft"]:
                #     messages = [
                #         {"role": "system", "content": instruction},
                #         {"role": "user", "content": prompt},
                #     ]
                #     res = predict(messages, model, tokenizer)
                # else:
                response = client.chat.completions.create(
                    model=model_name,
                    max_tokens=256,
                    timeout=60,
                    messages=[
                        {"role": "system", "content": instruction1},
                        {"role": "user", "content": prompt1},
                    ],
                    **cloud_kwargs
                )
                res1 = response.choices[0].message.content

                response = client.chat.completions.create(
                    model=model_name,
                    max_tokens=64,
                    timeout=60,
                    messages=[
                        {"role": "system", "content": instruction2},
                        {"role": "user", "content": "----------Question----------\n" + item["question"] + "\n\n----------Steps----------\n" + res1},
                    ],
                    **cloud_kwargs
                )
                res = response.choices[0].message.content
                break
            except Exception as e:
                res1 = ""
                res = ""
                print(f"[WARN] normal LLM call failed (attempt {attempt+1}): {e!r}", flush=True)

        return idx, {
            "question": item["question"],
            "answer": item["answers"][0],
            "res1": res1,
            "pred": res
        }

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, answer in tqdm(executor.map(process_item, range(len(dev)), dev), total=len(dev)):
            results.append((idx, answer))
    results.sort(key=lambda x: x[0])
    answers = [a for _, a in results]

    if use_best_hit and model in ("Qwen3.5-9B", "qwen3.5-9b"):
        with open(f"data/{dataset}/normal_pred_qwen35_9b_sep_use_best_hit{tag_suffix}.json", "w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit and model == "Qwen2.5-7B":
        with open(f"data/{dataset}/normal_pred_7b_sep_use_best_hit{tag_suffix}.json","w") as f:
            json.dump(answers, f, indent=4)
    # elif use_best_hit and model == "Qwen2.5-7B-ft":
    #     with open(f"data/{dataset}/normal_pred_7bft_sep_use_best_hit.json","w") as f:
    #         json.dump(answers, f, indent=4)
    elif use_best_hit and model == "Qwen2.5-0.5B":
        with open(f"data/{dataset}/normal_pred_05b_sep_use_best_hit.json","w") as f:
            json.dump(answers, f, indent=4)
    # elif use_best_hit and model == "Qwen2.5-0.5B-ft":
    #     with open(f"data/{dataset}/normal_pred_05bft_sep_use_best_hit.json","w") as f:
    #         json.dump(answers, f, indent=4)
    elif use_best_hit == False and model in ("Qwen3.5-9B", "qwen3.5-9b"):
        with open(f"data/{dataset}/normal_pred_qwen35_9b_sep.json", "w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit == False and model == "Qwen2.5-7B":
        with open(f"data/{dataset}/normal_pred_7b_sep.json","w") as f:
            json.dump(answers, f, indent=4)
    # elif use_best_hit == False and model == "Qwen2.5-7B-ft":
    #     with open(f"data/{dataset}/normal_pred_7bft_sep.json","w") as f:
    #         json.dump(answers, f, indent=4)
    elif use_best_hit and model == "Qwen2.5-72B":
        with open(f"data/{dataset}/normal_pred_72b_sep_use_best_hit.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit == False and model == "Qwen2.5-72B":
        with open(f"data/{dataset}/normal_pred_72b_sep.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit and model == "llama3-8B":
        with open(f"data/{dataset}/normal_pred_llama3_8b_sep_use_best_hit.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit == False and model == "llama3-8B":
        with open(f"data/{dataset}/normal_pred_llama3_8B_sep.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit and model == "llama3-70B":
        with open(f"data/{dataset}/normal_pred_llama3_70b_sep_use_best_hit.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit == False and model == "llama3-70B":
        with open(f"data/{dataset}/normal_pred_llama3_70B_sep.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit and model == "deepseek-r1":
        with open(f"data/{dataset}/normal_pred_ds_r1_sep_use_best_hit.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit == False and model == "deepseek-r1":
        with open(f"data/{dataset}/normal_pred_ds_r1_sep.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit and model == "deepseek-v3":
        with open(f"data/{dataset}/normal_pred_ds_v3_sep_use_best_hit.json","w") as f:
            json.dump(answers, f, indent=4)
    elif use_best_hit == False and model == "deepseek-v3":
        with open(f"data/{dataset}/normal_pred_ds_v3_sep.json","w") as f:
            json.dump(answers, f, indent=4)

    eval_res = evaluate(answers)

    if use_best_hit and model in ("Qwen3.5-9B", "qwen3.5-9b"):
        with open(f"data/{dataset}/eval_normal_pred_qwen35_9b_sep_use_best_hit{tag_suffix}.json", "w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit and model == "Qwen2.5-7B":
        with open(f"data/{dataset}/eval_normal_pred_7b_sep_use_best_hit{tag_suffix}.json","w") as f:
            json.dump(eval_res, f, indent=4)
    # elif use_best_hit and model == "Qwen2.5-7B-ft":
    #     with open(f"data/{dataset}/eval_normal_pred_7bft_sep_use_best_hit.json","w") as f:
    #         json.dump(eval_res, f, indent=4)
    elif use_best_hit and model == "Qwen2.5-0.5B":
        with open(f"data/{dataset}/eval_normal_pred_05b_sep_use_best_hit.json","w") as f:
            json.dump(eval_res, f, indent=4)
    # elif use_best_hit and model == "Qwen2.5-0.5B-ft":
    #     with open(f"data/{dataset}/eval_normal_pred_05bft_sep_use_best_hit.json","w") as f:
    #         json.dump(eval_res, f, indent=4)
    elif use_best_hit == False and model in ("Qwen3.5-9B", "qwen3.5-9b"):
        with open(f"data/{dataset}/eval_normal_pred_qwen35_9b_sep.json", "w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit == False and model == "Qwen2.5-7B":
        with open(f"data/{dataset}/eval_normal_pred_7b_sep.json","w") as f:
            json.dump(eval_res, f, indent=4)
    # elif use_best_hit == False and model == "Qwen2.5-7B-ft":
    #     with open(f"data/{dataset}/eval_normal_pred_7bft_sep.json","w") as f:
    #         json.dump(eval_res, f, indent=4)
    elif use_best_hit and model == "Qwen2.5-72B":
        with open(f"data/{dataset}/eval_normal_pred_72b_sep_use_best_hit.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit == False and model == "Qwen2.5-72B":
        with open(f"data/{dataset}/eval_normal_pred_72b_sep.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit and model == "llama3-8B":
        with open(f"data/{dataset}/eval_normal_pred_llama3_8B_sep_use_best_hit.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit == False and model == "llama3-8B":
        with open(f"data/{dataset}/eval_normal_pred_llama3_8B_sep.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit and model == "llama3-70B":
        with open(f"data/{dataset}/eval_normal_pred_llama3_70B_sep_use_best_hit.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit == False and model == "llama3-70B":
        with open(f"data/{dataset}/eval_normal_pred_llama3_70B_sep.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit and model == "deepseek-r1":
        with open(f"data/{dataset}/eval_normal_pred_ds_r1_sep_use_best_hit.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit == False and model == "deepseek-r1":
        with open(f"data/{dataset}/eval_normal_pred_ds_r1_sep.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit and model == "deepseek-v3":
        with open(f"data/{dataset}/eval_normal_pred_ds_v3_sep_use_best_hit.json","w") as f:
            json.dump(eval_res, f, indent=4)
    elif use_best_hit == False and model == "deepseek-v3":
        with open(f"data/{dataset}/eval_normal_pred_ds_v3_sep.json","w") as f:
            json.dump(eval_res, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--model", default="Qwen3.5-9B", type=str)
    parser.add_argument("--use_best_hit", action="store_true")
    parser.add_argument("--embed_model", default="BAAI/bge-base-en-v1.5", type=str)
    parser.add_argument("--rerank_model", default="BAAI/bge-reranker-large", type=str)
    parser.add_argument("--max_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--tag", default="", type=str)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--no_thinking", action="store_true")
    args = parser.parse_args()

    normal_predict(args.dataset, args.use_best_hit, args.model, args.embed_model, args.rerank_model, args.max_workers, args.device, tag=args.tag, limit=args.limit, no_thinking=args.no_thinking)
