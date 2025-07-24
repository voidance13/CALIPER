import random
import json
import os
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import fasttext
import argparse

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
    

def hybrid_predict(dataset, use_best_hit=False, retrieve_step_by_step=False, small_model="Qwen2.5-7B", large_model="Qwen2.5-72B"):

    if large_model == "Qwen2.5-72B":
        # generate_steps_client = OpenAI(
        #     api_key="vllm",
        #     base_url="http://localhost:6006/v1",
        # )
        generate_steps_client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key="",
            base_url="",
        )
        large_model_name = "qwen2.5-72b-instruct"
    elif large_model == "deepseek-r1" or large_model == "deepseek-v3":
        generate_steps_client = OpenAI(
            api_key="",
            base_url="",
        )

    if use_best_hit:
        with open(f"data/{dataset}/test.json") as f:
            dev = json.load(f)
    else:
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
        if dataset == "news_articles":
            with open(f"data/{dataset}/test.json") as f:
                questions = json.load(f)
            with open(f"data/{dataset}/corpus.json") as f:
                corpus = json.load(f)
            text = ""
            for article in corpus:
                text += article["title"] + "\n\n" + article["body"] + "\n\n"
            text = text[:-2]
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
                chunk_size=400,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
            )
            texts = [doc.page_content for doc in text_splitter.create_documents([text])]
            faiss_vectorstore = FAISS.from_texts(
                texts=texts, embedding=embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )
            faiss_retriever = faiss_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20, "score_threshold": 0.6})
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=faiss_retriever
            )
            retrieved_result = []
            for q in tqdm(questions):
                retrieved_text = ""
                for srcdoc in retriever.get_relevant_documents(q["query"]):
                    retrieved_text += srcdoc.page_content + "\n\n"
                q["retrieved_text"] = retrieved_text[:-2]
                retrieved_result.append(q)
        else:
            with open(f"data/{dataset}/test.json") as f:
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

    fasttext_model = fasttext.load_model('intent_detection/train/train_bi.bin')

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

        question = item["question"]
        pred = fasttext_model.predict(question)[0]
        
        if "__label__easy" in pred:
            instruction1 = "You are a helpful assistant. According to given articles, answer the given question. Think step by step. Output the executing result of every step."
            prompt1 = "Question:\n" + item["question"] + "\n\nCorpus:\n" + text
            instruction2 = "According to the given question and the reasoning steps, output the complete final answer as short as possible. Specially, if the given question is a yes-no question, just output \"yes\" or \"no\"."
            
            try:
                response = client.chat.completions.create(
                    model=small_model,
                    messages=[
                        {"role": "system", "content": instruction1},
                        {"role": "user", "content": prompt1},
                    ],
                )
                res1 = response.choices[0].message.content

                response = client.chat.completions.create(
                    model=small_model,
                    messages=[
                        {"role": "system", "content": instruction2},
                        {"role": "user", "content": "----------Question----------\n" + item["question"] + "\n\n----------Steps----------\n" + res1},
                    ],
                )
                res = response.choices[0].message.content
            except:
                res = ""
        else:
            steps = generate_steps(generate_steps_client, large_model_name, question)

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
            with open(f"data/{dataset}/hybrid_pred_7b_72b_sep_clf_use_best_hit.json","w") as f:
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
    parser.add_argument("--retrieve_step_by_step", action="store_true")
    args = parser.parse_args()

    hybrid_predict(args.dataset, args.use_best_hit, args.retrieve_step_by_step, args.small_model, args.large_model)