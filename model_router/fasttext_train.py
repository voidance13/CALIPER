import fasttext
import random
import jieba
import json
import re

stopwords = {word.strip() for word in open('model_router/stopwords.txt', encoding='utf-8')}


def clean_text(text):
    segs = jieba.lcut(text)
    segs = list(filter(lambda x: len(x) > 1, segs))
    segs = list(filter(lambda x: x not in stopwords, segs))
    return " ".join(segs)


def preprocess(dataset):
    with open("data/hotpotqa/train.json") as f:
        hotpotqa = json.load(f)
    with open("data/2wikimultihopqa/train.json") as f:
        wikimultihopqa = json.load(f)
    with open("data/musique/train.json") as f:
        musique = json.load(f)

    data_list = []

    for sample in hotpotqa[:10000]:
        if dataset == "hotpotqa":
            data_list.append("__label__1 " + clean_text(sample["question"]) + "\n")
        else:
            data_list.append("__label__0 " + clean_text(sample["question"]) + "\n")
    for sample in wikimultihopqa[:10000]:
        if dataset == "2wikimultihopqa":
            data_list.append("__label__1 " + clean_text(sample["question"]) + "\n")
        else:
            data_list.append("__label__0 " + clean_text(sample["question"]) + "\n")
    for sample in musique[:10000]:
        if dataset == "musique":
            data_list.append("__label__1 " + clean_text(sample["question"]) + "\n")
        else:
            data_list.append("__label__0 " + clean_text(sample["question"]) + "\n")
    
    size = len(data_list)
    train_data = random.sample(data_list, int(size * 0.8))
    valid_data = [i for i in data_list if i not in train_data]

    with open("data/hotpotqa/test.json") as f:
        hotpotqa_test = json.load(f)
    with open("data/2wikimultihopqa/test.json") as f:
        wikimultihopqa_test = json.load(f)
    with open("data/musique/test.json") as f:
        musique_test = json.load(f)
    
    hotpotqa_test_data = []
    wikimultihopqa_test_data = []
    musique_test_data = []

    for sample in hotpotqa_test[:200]:
        if dataset == "hotpotqa":
            hotpotqa_test_data.append("__label__1 " + clean_text(sample["question"]) + "\n")
        else:
            hotpotqa_test_data.append("__label__0 " + clean_text(sample["question"]) + "\n")
    for sample in wikimultihopqa_test[:200]:
        if dataset == "2wikimultihopqa":
            wikimultihopqa_test_data.append("__label__1 " + clean_text(sample["question"]) + "\n")
        else:
            wikimultihopqa_test_data.append("__label__0 " + clean_text(sample["question"]) + "\n")
    for sample in musique_test[:200]:
        if dataset == "musique":
            musique_test_data.append("__label__1 " + clean_text(sample["question"]) + "\n")
        else:
            musique_test_data.append("__label__0 " + clean_text(sample["question"]) + "\n")

    open(f"model_router/train/{dataset}_train.txt", 'w', encoding='utf-8').writelines(train_data)
    open(f"model_router/train/{dataset}_valid.txt", 'w', encoding='utf-8').writelines(valid_data)

    total = hotpotqa_test_data
    total.extend(wikimultihopqa_test_data)
    total.extend(musique_test_data)
    open(f"model_router/train/{dataset}_test_total600.txt", 'w', encoding='utf-8').writelines(total)


def train(dataset):
    model = fasttext.train_supervised(input=f"model_router/train/{dataset}_train.txt",
                                      autotuneValidationFile=f'model_router/train/{dataset}_valid.txt',
                                      autotuneDuration=10,
                                    #   autotuneModelSize="60M",
                                      autotuneMetric='f1',
                                      verbose=3)

    model.save_model(f'model_router/train/{dataset}_train.bin')


def test(dataset):
    model = fasttext.load_model(f'model_router/train/{dataset}_train.bin')

    samples, inputs = [], []
    for line in open(f'model_router/train/{dataset}_test_total600.txt', encoding='utf-8'):
        sample = line.split()
        samples.append(sample)
        inputs.append(' '.join(sample[1:]))
    preds = model.predict(inputs)[0]
    print(len(preds))
    right = 0
    for sample, pred in zip(samples, preds):
        if sample[0] in pred:
            right += 1
            # print(sample)

    print(right/len(preds))


if __name__ == "__main__":
    dataset = "musique"
    preprocess(dataset)
    train(dataset)
    test(dataset)