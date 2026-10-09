import fasttext
import random
import jieba
import json
import re

stopwords = {word.strip() for word in open('query_clf/stopwords.txt', encoding='utf-8')}


def clean_text(text):
    segs = jieba.lcut(text)
    segs = list(filter(lambda x: len(x) > 1, segs))
    segs = list(filter(lambda x: x not in stopwords, segs))
    return " ".join(segs)


def preprocess():
    with open("query_clf/train/hotpotqa10000.json") as f:
        hotpotqa = json.load(f)
    with open("query_clf/train/musique10000.json") as f:
        musique = json.load(f)

    data_list = []

    for sample in hotpotqa:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # data_list.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"] + text) + "\n")
        data_list.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"]) + "\n")
    for sample in musique:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # data_list.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"] + text) + "\n")
        data_list.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"]) + "\n")
    
    size = len(data_list)
    print(size)
    train_data = random.sample(data_list, int(size * 0.8))
    valid_data = [i for i in data_list if i not in train_data]

    with open("query_clf/train/hotpotqa_test2000_easy.json") as f:
        hotpotqa_test2000_easy = json.load(f)
    with open("query_clf/train/hotpotqa_test2000_middle.json") as f:
        hotpotqa_test2000_middle = json.load(f)
    with open("query_clf/train/hotpotqa_test2000_hard.json") as f:
        hotpotqa_test2000_hard = json.load(f)
    with open("query_clf/train/musique_test2000_easy.json") as f:
        musique_test2000_easy = json.load(f)
    with open("query_clf/train/musique_test2000_middle.json") as f:
        musique_test2000_middle = json.load(f)
    with open("query_clf/train/musique_test2000_hard.json") as f:
        musique_test2000_hard = json.load(f)

    hotpotqa_test_easy = []
    hotpotqa_test_middle = []
    hotpotqa_test_hard = []
    musique_test_easy = []
    musique_test_middle = []
    musique_test_hard = []

    for sample in hotpotqa_test2000_easy:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # hotpotqa_test_easy.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"] + text) + "\n")
        hotpotqa_test_easy.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"]) + "\n")
    for sample in hotpotqa_test2000_middle:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # hotpotqa_test_middle.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"] + text) + "\n")
        hotpotqa_test_middle.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"]) + "\n")
    for sample in hotpotqa_test2000_hard:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # hotpotqa_test_hard.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"] + text) + "\n")
        hotpotqa_test_hard.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"]) + "\n")
    for sample in musique_test2000_easy:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # musique_test_easy.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"] + text) + "\n")
        musique_test_easy.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"]) + "\n")
    for sample in musique_test2000_middle:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # musique_test_middle.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"] + text) + "\n")
        musique_test_middle.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"]) + "\n")
    for sample in musique_test2000_hard:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # musique_test_hard.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"] + text) + "\n")
        musique_test_hard.append("__label__" + sample["intent_type"] + " " + clean_text(sample["question"]) + "\n")

    open("query_clf/train/train.txt", 'w', encoding='utf-8').writelines(train_data)
    open("query_clf/train/valid.txt", 'w', encoding='utf-8').writelines(valid_data)
    open("query_clf/train/hotpotqa_test2000_easy.txt", 'w', encoding='utf-8').writelines(hotpotqa_test_easy)
    open("query_clf/train/hotpotqa_test2000_middle.txt", 'w', encoding='utf-8').writelines(hotpotqa_test_middle)
    open("query_clf/train/hotpotqa_test2000_hard.txt", 'w', encoding='utf-8').writelines(hotpotqa_test_hard)
    open("query_clf/train/musique_test2000_easy.txt", 'w', encoding='utf-8').writelines(musique_test_easy)
    open("query_clf/train/musique_test2000_middle.txt", 'w', encoding='utf-8').writelines(musique_test_middle)
    open("query_clf/train/musique_test2000_hard.txt", 'w', encoding='utf-8').writelines(musique_test_hard)

    total = hotpotqa_test_easy
    total.extend(hotpotqa_test_middle)
    total.extend(hotpotqa_test_hard)
    total.extend(musique_test_easy)
    total.extend(musique_test_middle)
    total.extend(musique_test_hard)
    open("query_clf/train/test_total.txt", 'w', encoding='utf-8').writelines(total)


def preprocess_bi():
    with open("query_clf/train/hotpotqa10000.json") as f:
        hotpotqa = json.load(f)
    with open("query_clf/train/musique10000.json") as f:
        musique = json.load(f)

    data_list = []

    for sample in hotpotqa:
        if sample["intent_type"] == "easy":
            # text = ""
            # ctxs = []
            # for ctx in sample["ctxs"]:
            #     ctxs.append(ctx["sentences"])
            # for supporting_fact in sample["supporting_facts"]:
            #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
            # data_list.append("__label__easy " + clean_text(sample["question"] + text) + "\n")
            data_list.append("__label__easy " + clean_text(sample["question"]) + "\n")
        else:
            # text = ""
            # ctxs = []
            # for ctx in sample["ctxs"]:
            #     ctxs.append(ctx["sentences"])
            # for supporting_fact in sample["supporting_facts"]:
            #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
            # data_list.append("__label__hard " + clean_text(sample["question"] + text) + "\n")
            data_list.append("__label__hard " + clean_text(sample["question"]) + "\n")
    for sample in musique:
        if sample["intent_type"] == "easy":
            # text = ""
            # ctxs = []
            # for ctx in sample["ctxs"]:
            #     ctxs.append(ctx["sentences"])
            # for supporting_fact in sample["supporting_facts"]:
            #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
            # data_list.append("__label__easy " + clean_text(sample["question"] + text) + "\n")
            data_list.append("__label__easy " + clean_text(sample["question"]) + "\n")
        else:
            # text = ""
            # ctxs = []
            # for ctx in sample["ctxs"]:
            #     ctxs.append(ctx["sentences"])
            # for supporting_fact in sample["supporting_facts"]:
            #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
            # data_list.append("__label__hard " + clean_text(sample["question"] + text) + "\n")
            data_list.append("__label__hard " + clean_text(sample["question"]) + "\n")
    
    size = len(data_list)
    print(size)
    train_data = random.sample(data_list, int(size * 0.8))
    valid_data = [i for i in data_list if i not in train_data]

    with open("query_clf/train/hotpotqa_test2000_easy.json") as f:
        hotpotqa_test2000_easy = json.load(f)
    with open("query_clf/train/hotpotqa_test2000_middle.json") as f:
        hotpotqa_test2000_middle = json.load(f)
    with open("query_clf/train/hotpotqa_test2000_hard.json") as f:
        hotpotqa_test2000_hard = json.load(f)
    with open("query_clf/train/musique_test2000_easy.json") as f:
        musique_test2000_easy = json.load(f)
    with open("query_clf/train/musique_test2000_middle.json") as f:
        musique_test2000_middle = json.load(f)
    with open("query_clf/train/musique_test2000_hard.json") as f:
        musique_test2000_hard = json.load(f)

    hotpotqa_test_easy = []
    hotpotqa_test_hard = []
    musique_test_easy = []
    musique_test_hard = []

    for sample in hotpotqa_test2000_easy:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # hotpotqa_test_easy.append("__label__easy " + clean_text(sample["question"] + text) + "\n")
        hotpotqa_test_easy.append("__label__easy " + clean_text(sample["question"]) + "\n")
    for sample in hotpotqa_test2000_middle:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # hotpotqa_test_hard.append("__label__hard " + clean_text(sample["question"] + text) + "\n")
        hotpotqa_test_hard.append("__label__hard " + clean_text(sample["question"]) + "\n")
    for sample in hotpotqa_test2000_hard:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # hotpotqa_test_hard.append("__label__hard " + clean_text(sample["question"] + text) + "\n")
        hotpotqa_test_hard.append("__label__hard " + clean_text(sample["question"]) + "\n")
    for sample in musique_test2000_easy:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # musique_test_easy.append("__label__easy " + clean_text(sample["question"] + text) + "\n")
        musique_test_easy.append("__label__easy " + clean_text(sample["question"]) + "\n")
    for sample in musique_test2000_middle:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # musique_test_hard.append("__label__hard " + clean_text(sample["question"] + text) + "\n")
        musique_test_hard.append("__label__hard " + clean_text(sample["question"]) + "\n")
    for sample in musique_test2000_hard:
        # text = ""
        # ctxs = []
        # for ctx in sample["ctxs"]:
        #     ctxs.append(ctx["sentences"])
        # for supporting_fact in sample["supporting_facts"]:
        #     text += ctxs[supporting_fact[0]][supporting_fact[1]] + "\n"
        # musique_test_hard.append("__label__hard " + clean_text(sample["question"] + text) + "\n")
        musique_test_hard.append("__label__hard " + clean_text(sample["question"]) + "\n")

    open("query_clf/train/train_bi.txt", 'w', encoding='utf-8').writelines(train_data)
    open("query_clf/train/valid_bi.txt", 'w', encoding='utf-8').writelines(valid_data)
    open("query_clf/train/hotpotqa_test2000_easy_bi.txt", 'w', encoding='utf-8').writelines(hotpotqa_test_easy)
    open("query_clf/train/hotpotqa_test2000_hard_bi.txt", 'w', encoding='utf-8').writelines(hotpotqa_test_hard)
    open("query_clf/train/musique_test2000_easy_bi.txt", 'w', encoding='utf-8').writelines(musique_test_easy)
    open("query_clf/train/musique_test2000_hard_bi.txt", 'w', encoding='utf-8').writelines(musique_test_hard)

    total = hotpotqa_test_easy
    total.extend(hotpotqa_test_hard)
    total.extend(musique_test_easy)
    total.extend(musique_test_hard)
    open("query_clf/train/test_total_bi.txt", 'w', encoding='utf-8').writelines(total)


def train():
    model = fasttext.train_supervised(input="query_clf/train/train.txt",
                                      autotuneValidationFile='query_clf/train/valid.txt',
                                      autotuneDuration=10,
                                    #   autotuneModelSize="60M",
                                      autotuneMetric='f1',
                                      verbose=3)

    model.save_model('query_clf/train/train.bin')


def train_bi():
    model = fasttext.train_supervised(input="query_clf/train/train_bi.txt",
                                      autotuneValidationFile='query_clf/train/valid_bi.txt',
                                      autotuneDuration=10,
                                    #   autotuneModelSize="60M",
                                      autotuneMetric='f1',
                                      verbose=3)

    model.save_model('query_clf/train/train_bi.bin')


def test():
    model = fasttext.load_model('query_clf/train/train.bin')

    labels, inputs = [], []
    for line in open('query_clf/train/test_total.txt', encoding='utf-8'):
        line = line.split()
        labels.append(line[0])
        inputs.append(' '.join(line[1:]))
    preds = model.predict(inputs)[0]
    print(len(preds))
    right = 0
    easy_right = 0
    middle_right = 0
    hard_right = 0
    for label, pred in zip(labels, preds):
        if label in pred:
            right += 1
            if label == '__label__easy':
                easy_right += 1
            if label == '__label__middle':
                middle_right += 1
            if label == '__label__hard':
                hard_right += 1
        if label != '__label__easy' and '__label__easy' not in pred:
            easy_right += 1
        if label != '__label__middle' and '__label__middle' not in pred:
            middle_right += 1
        if label != '__label__hard' and '__label__hard' not in pred:
            hard_right += 1
            
        
    print(right/len(preds))
    print(easy_right/len(preds))
    print(middle_right/len(preds))
    print(hard_right/len(preds))


def test_bi():
    model = fasttext.load_model('query_clf/train/train_bi.bin')

    samples, inputs = [], []
    for line in open('query_clf/train/test_total_bi.txt', encoding='utf-8'):
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
    # preprocess()
    # train()
    # test()
    preprocess_bi()
    train_bi()
    test_bi()