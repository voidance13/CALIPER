import json
from openai import OpenAI
from tqdm import tqdm
import argparse
import dashscope


def generate_steps(dataset, large_model):

    with open(f"data/{dataset}/test.json") as f:
        dev = json.load(f)

    if large_model == "Qwen2.5-72B":
        client = OpenAI(
            api_key="vllm",
            base_url="http://localhost:6006/v1",
        )
    elif large_model == "deepseek-r1" or large_model == "deepseek-v3":
        # client = OpenAI(
        #     api_key="sk-609217c7b16a49c7a73e2752906f3acc",
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # )
        client = OpenAI(
            api_key="sk-jnomkPZiYo4qHsaGXf1aCniwT71lcENjI3CiUH5IU1OlCHFc",
            base_url="https://api.lkeap.cloud.tencent.com/v1",
        )

    result = []

    for sample in tqdm(dev):
        if dataset == "news_articles":
            question = sample["query"]
        else:
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
        
    if large_model == "Qwen2.5-72B":
        with open(f"data/{dataset}/test_with_72b_generate_steps.json","w") as f:
            json.dump(result, f, indent=4)
    elif large_model == "deepseek-r1":
        with open(f"data/{dataset}/test_with_ds_r1_generate_steps.json","w") as f:
            json.dump(result, f, indent=4)
    elif large_model == "deepseek-v3":
        with open(f"data/{dataset}/test_with_ds_v3_generate_steps.json","w") as f:
            json.dump(result, f, indent=4)


def generate_steps_simple(dataset, large_model):

    with open(f"data/{dataset}/test.json") as f:
        dev = json.load(f)

    if large_model == "Qwen2.5-72B":
        # client = OpenAI(
        #     api_key="vllm",
        #     base_url="http://localhost:6006/v1",
        # )
        client = OpenAI(
            api_key="",
            base_url="",
        )
        large_model_name = "qwen2.5-72b-instruct"

    elif large_model == "deepseek-r1" or large_model == "deepseek-v3":
        client = OpenAI(
            api_key="",
            base_url="",
        )

    result = []

    for sample in tqdm(dev):
        question = sample["question"]
        
        try:
            response = client.chat.completions.create(
                model=large_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Output the solution steps for the given question. Don't use any information that is not given. Don't give any example. Only generate necessary steps succinctly."},
                    {"role": "user", "content": question},
                ]
            )
            res = response.choices[0].message.content
            # print(res)
        except:
            res = ""

        sample["72b_generate_steps"] = res
        result.append(sample)
        
    if large_model == "Qwen2.5-72B":
        with open(f"data/{dataset}/test_with_72b_generate_steps_simple.json","w") as f:
            json.dump(result, f, indent=4)
    elif large_model == "deepseek-r1":
        with open(f"data/{dataset}/test_with_ds_r1_generate_steps_simple.json","w") as f:
            json.dump(result, f, indent=4)
    elif large_model == "deepseek-v3":
        with open(f"data/{dataset}/test_with_ds_v3_generate_steps_simple.json","w") as f:
            json.dump(result, f, indent=4)