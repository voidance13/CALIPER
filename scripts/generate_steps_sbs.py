import json
from openai import OpenAI
from tqdm import tqdm
import argparse
import re


def generate_steps_sbs(dataset, large_model):

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

    for sample in tqdm(dev):
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

        steps = re.split("\n*[1-9][0-9]*\. ?\n*", res)[1:]

        sample["72b_generate_steps"] = steps
        result.append(sample)
        
    if large_model == "Qwen2.5-72B":
        with open(f"data/{dataset}/test_with_72b_generate_steps_sbs.json","w") as f:
            json.dump(result, f, indent=4)
    elif large_model == "deepseek-r1":
        with open(f"data/{dataset}/test_with_ds_r1_generate_steps_sbs.json","w") as f:
            json.dump(result, f, indent=4)
    elif large_model == "deepseek-v3":
        with open(f"data/{dataset}/test_with_ds_v3_generate_steps_sbs.json","w") as f:
            json.dump(result, f, indent=4)