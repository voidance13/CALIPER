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
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
aliyun_api_key = os.getenv("ALIYUN_API_KEY")
siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")

from evaluate import evaluate
from BCERerank import BCERerank


def generate_steps(client, model, question, no_thinking=False, max_tokens=256):
    try:
        create_kwargs = {}
        if no_thinking:
            create_kwargs["extra_body"] = {"enable_thinking": False}
        else:
            create_kwargs["extra_body"] = {"enable_thinking": True, "thinking_budget": 512}
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            timeout=60,
            messages=[
                {"role": "system", "content": "You are an instructor. Given the question, output a numbered list of instructions for solving it. Each instruction must be an action such as \"Identify...\", \"Find...\", \"Check...\" or \"Compare...\". Do NOT include any conclusions, facts, or answers in the instructions. Do not answer the question."},
                {"role": "user", "content": question},
            ],
            **create_kwargs
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


def load_cached_retrieval(retrieved_path):
    if os.path.exists(retrieved_path):
        with open(retrieved_path, encoding="utf-8") as f:
            print(f"Loading cached retrieval from {retrieved_path}")
            return json.load(f)
    return None


REFLECT_INSTRUCTION = (
    "You are an advanced reasoning agent that can improve based on self reflection. "
    "You will be given a question, relevant corpus and a previous answer. Check the answer. "
    "If the answer is reasonable, only output \"Correct\". Otherwise, diagnose a possible reason "
    "for failure and devise a new, concise, high level plan that aims to mitigate the same failure "
    "in a few sentences. Use complete sentences."
)

REFLECT_INSTRUCTION_FIX = (
    "You are an advanced reasoning agent that can improve based on self reflection. "
    "You will be given a question, reasoning steps, relevant corpus and a previous answer. "
    "Check whether the previous answer is consistent with the reasoning steps and the corpus. "
    "If it is consistent and reasonable, only output \"Correct\". Otherwise, diagnose a possible "
    "reason for failure and devise a new, concise, high level plan that aims to mitigate the same "
    "failure in a few sentences. Use complete sentences."
)


def reflect_answer(client, small_model_name, item, text, res1, current, instruction2, max_reflect_iter, fix_chain=False, no_thinking_reflection=False):
    reflection = ""
    iterations = 0
    for _ in range(max_reflect_iter):
        if fix_chain:
            sys_content = REFLECT_INSTRUCTION_FIX
            user_content = ("----------Steps----------\n" + res1 + "\n\n----------Corpus----------\n" + text
                            + "\n\n----------Question----------\n" + item["question"]
                            + "\n\n----------Previous answer----------\n" + current)
        else:
            sys_content = REFLECT_INSTRUCTION
            user_content = ("----------Corpus----------\n" + text
                            + "\n\n----------Question----------\n" + item["question"]
                            + "\n\n----------Previous answer----------\n" + current)
        create_kwargs = {}
        if no_thinking_reflection:
            create_kwargs["extra_body"] = {"enable_thinking": False}
        response = client.chat.completions.create(
            model=small_model_name,
            max_tokens=256,
            timeout=60,
            messages=[
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content},
            ],
            **create_kwargs
        )
        reflection = response.choices[0].message.content or ""
        if ("Correct" in reflection or "correct" in reflection) and "incorrect" not in reflection and "Incorrect" not in reflection:
            break
        iterations += 1
        response = client.chat.completions.create(
            model=small_model_name,
            max_tokens=(128 if fix_chain else 64),
            timeout=60,
            messages=[
                {"role": "system", "content": instruction2},
                {"role": "user", "content": "----------Steps----------\n" + res1 + "\n\n" + reflection + "\n\n----------Question----------\n" + item["question"]},
            ],
        )
        current = response.choices[0].message.content or ""
    return current, reflection, iterations


# w/o model router
def retrieve1(large_model, retrieved_path, embed_model, rerank_model, dataset, device="cuda:0"):
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


def retrieve2(large_model, retrieved_path, embed_model, rerank_model, dataset, device="cuda:0"):
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


# 
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


def hybrid_predict(dataset, use_best_hit=False, small_model="Qwen2.5-7B", large_model="Qwen2.5-72B", embed_model="BAAI/bge-base-en-v1.5", rerank_model="BAAI/bge-reranker-large", max_workers=8, device="cuda:0", use_reflection=False, max_reflect_iter=3, no_thinking=False, tag="", limit=0, steps_only=False, fix_chain=False, no_thinking_reflection=False):
    tag_suffix = f"_{tag}" if tag else ""
    
    if large_model == "Qwen2.5-72B":
        # generate_steps_client = OpenAI(
        #     api_key="vllm",
        #     base_url="http://localhost:6006/v1",
        # )
        generate_steps_client = OpenAI(
            api_key=aliyun_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        large_model_name = "qwen3.7-plus"
    elif large_model == "llama3-70B":
        generate_steps_client = OpenAI(
            api_key="vllm",
            base_url="http://localhost:6009/v1",
        )
        large_model_name = "llama3.1-70b-instruct"
    elif large_model == "deepseek-r1":
        generate_steps_client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com",
        )
        large_model_name = "deepseek-reasoner"
    elif large_model == "deepseek-v3":
        generate_steps_client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com",
        )
        large_model_name = "deepseek-chat"
        
    if use_best_hit:
        with open(f"data/{dataset}/test.json") as f:
            dev = json.load(f)
    else:
        retrieved_path = f"data/{dataset}/test_retrieved.json"

        dev = retrieve1(large_model, retrieved_path, embed_model, rerank_model, dataset, device=device)
        # dev = retrieve2(large_model, retrieved_path, embed_model, rerank_model, dataset)

    if limit > 0:
        dev = dev[:limit]

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
            
        steps = generate_steps(generate_steps_client, large_model_name, item["question"], no_thinking=no_thinking, max_tokens=(512 if fix_chain else 256))

        instruction1 = "You are a helpful assistant. According to the given steps and corpus, answer the given question. Output the executing result of every step in simple sentences."
        if steps_only:
            prompt = "----------Question----------\n" + item["question"] + "\n\n----------Steps----------\n" + steps
        else:
            prompt = "----------Question----------\n" + item["question"] + "----------Corpus----------\n" + text + "\n\n----------Steps----------\n" + steps + "\n\n----------Question----------\n" + item["question"]
        instruction2 = "Extract the final answer from the given reasoning steps. Output only the final answer, as short as possible."

        res1 = ""
        res = ""
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=small_model_name,
                    max_tokens=256,
                    timeout=60,
                    messages=[
                        {"role": "system", "content": instruction1},
                        {"role": "user", "content": prompt},
                    ],
                )
                res1 = response.choices[0].message.content

                response = client.chat.completions.create(
                    model=small_model_name,
                    max_tokens=(128 if fix_chain else 64),
                    timeout=60,
                    messages=[
                        {"role": "system", "content": instruction2},
                        {"role": "user", "content": "----------Steps----------\n" + res1 + "\n\n----------Question----------\n" + item["question"]},
                    ],
                )
                res = response.choices[0].message.content
                break
            except Exception as e:
                res1 = ""
                res = ""
                print(f"[WARN] hybrid local LLM call failed (attempt {attempt+1}): {e!r}", flush=True)

        pred = res
        reflection = ""
        reflect_iters = 0
        if use_reflection:
            pred, reflection, reflect_iters = reflect_answer(
                client, small_model_name, item, text, res1, res, instruction2, max_reflect_iter, fix_chain=fix_chain, no_thinking_reflection=no_thinking_reflection
            )

        return idx, {
            "question": item["question"],
            "answer": item["answers"][0],
            "steps": steps,
            "res1": res1,
            "reflection": reflection,
            "reflection_iterations": reflect_iters,
            "pred": pred
        }

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, answer in tqdm(executor.map(process_item, range(len(dev)), dev), total=len(dev)):
            results.append((idx, answer))
    results.sort(key=lambda x: x[0])
    answers = [a for _, a in results]

    if use_best_hit:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/hybrid_pred_7b_72b_sep_use_best_hit{tag_suffix}.json","w") as f:
                json.dump(answers, f, indent=4)
        if small_model == "llama3-8B" and large_model == "llama3-70B":
            with open(f"data/{dataset}/hybrid_pred_llama3_8b_70b_sep_use_best_hit.json","w") as f:
                json.dump(answers, f, indent=4)
        if small_model == "llama3-8B" and large_model == "deepseek-r1":
            with open(f"data/{dataset}/hybrid_pred_llama3_8b_ds_r1_sep_use_best_hit.json","w") as f:
                json.dump(answers, f, indent=4)
    else:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/hybrid_pred_7b_72b_sep.json","w") as f:
                json.dump(answers, f, indent=4)
        if small_model == "llama3-8B" and large_model == "llama3-70B":
            with open(f"data/{dataset}/hybrid_pred_llama3_8b_70b_sep.json","w") as f:
                json.dump(answers, f, indent=4)
        if small_model == "llama3-8B" and large_model == "deepseek-r1":
            with open(f"data/{dataset}/hybrid_pred_llama3_8b_ds_r1_sep.json","w") as f:
                json.dump(answers, f, indent=4)

    eval_res = evaluate(answers)

    if use_best_hit:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_72b_sep_use_best_hit{tag_suffix}.json","w") as f:
                json.dump(eval_res, f, indent=4)
        if small_model == "llama3-8B" and large_model == "llama3-70B":
            with open(f"data/{dataset}/eval_hybrid_pred_llama3_8b_70b_sep_use_best_hit.json","w") as f:
                json.dump(answers, f, indent=4)
        if small_model == "llama3-8B" and large_model == "deepseek-r1":
            with open(f"data/{dataset}/eval_hybrid_pred_llama3_8b_ds_r1_sep_use_best_hit.json","w") as f:
                json.dump(answers, f, indent=4)
    else:
        if small_model == "Qwen2.5-7B" and large_model == "Qwen2.5-72B":
            with open(f"data/{dataset}/eval_hybrid_pred_7b_72b_sep.json","w") as f:
                json.dump(eval_res, f, indent=4)
        if small_model == "llama3-8B" and large_model == "llama3-70B":
            with open(f"data/{dataset}/eval_hybrid_pred_llama3_8b_70b_sep.json","w") as f:
                json.dump(answers, f, indent=4)
        if small_model == "llama3-8B" and large_model == "deepseek-r1":
            with open(f"data/{dataset}/eval_hybrid_pred_llama3_8b_ds_r1_sep.json","w") as f:
                json.dump(answers, f, indent=4)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--small_model", default="Qwen2.5-7B", type=str)
    parser.add_argument("--large_model", default="Qwen2.5-72B", type=str)
    parser.add_argument("--use_best_hit", action="store_true")
    parser.add_argument("--embed_model", default="BAAI/bge-base-en-v1.5", type=str)
    parser.add_argument("--rerank_model", default="BAAI/bge-reranker-large", type=str)
    parser.add_argument("--max_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--use_reflection", action="store_true")
    parser.add_argument("--max_reflect_iter", default=3, type=int)
    parser.add_argument("--no_thinking", action="store_true")
    parser.add_argument("--no_thinking_reflection", action="store_true")
    parser.add_argument("--tag", default="", type=str)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--steps_only", action="store_true")
    parser.add_argument("--fix_chain", action="store_true")
    args = parser.parse_args()

    hybrid_predict(args.dataset, args.use_best_hit, args.small_model, args.large_model, args.embed_model, args.rerank_model, args.max_workers, args.device, args.use_reflection, args.max_reflect_iter, no_thinking=args.no_thinking, tag=args.tag, limit=args.limit, steps_only=args.steps_only, fix_chain=args.fix_chain, no_thinking_reflection=args.no_thinking_reflection)
