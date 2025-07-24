import json


# with open("data/hotpotqa/test.json") as f:
#     train = json.load(f)

# with open("intent_detection/easy_hotpotqa_test2000.json") as f:
#     easy_list = json.load(f)
# with open("intent_detection/hard_hotpotqa_test2000.json") as f:
#     hard_list = json.load(f)

# easy = []
# middle = []
# hard = []

# for sample in train[:2000]:
#     if sample["id"] in easy_list:
#         sample["intent_type"] = "easy"
#         easy.append(sample)
#     elif sample["id"] in hard_list:
#         sample["intent_type"] = "hard"
#         hard.append(sample)
#     else:
#         sample["intent_type"] = "middle"
#         middle.append(sample)

# with open("intent_detection/train/hotpotqa_test2000_easy.json", "w") as f:
#     json.dump(easy, f, indent=4)
# with open("intent_detection/train/hotpotqa_test2000_middle.json", "w") as f:
#     json.dump(middle, f, indent=4)
# with open("intent_detection/train/hotpotqa_test2000_hard.json", "w") as f:
#     json.dump(hard, f, indent=4)


with open("data/musique/test.json") as f:
    train = json.load(f)

with open("intent_detection/easy_musique_test2000.json") as f:
    easy_list = json.load(f)
with open("intent_detection/hard_musique_test2000.json") as f:
    hard_list = json.load(f)

easy = []
middle = []
hard = []

for sample in train[:2000]:
    if sample["id"] in easy_list:
        sample["intent_type"] = "easy"
        easy.append(sample)
    elif sample["id"] in hard_list:
        sample["intent_type"] = "hard"
        hard.append(sample)
    else:
        sample["intent_type"] = "middle"
        middle.append(sample)

with open("intent_detection/train/musique_test2000_easy.json", "w") as f:
    json.dump(easy, f, indent=4)
with open("intent_detection/train/musique_test2000_middle.json", "w") as f:
    json.dump(middle, f, indent=4)
with open("intent_detection/train/musique_test2000_hard.json", "w") as f:
    json.dump(hard, f, indent=4)


# with open("data/2wikimultihopqa/train.json") as f:
#     train = json.load(f)

# with open("intent_detection/easy_2wikimultihopqa2000.json") as f:
#     easy = json.load(f)
# with open("intent_detection/hard_2wikimultihopqa2000.json") as f:
#     hard = json.load(f)

# result = []

# for sample in train[:2000]:
#     if sample["id"] in easy:
#         sample["intent_type"] = "easy"
#     elif sample["id"] in hard:
#         sample["intent_type"] = "hard"
#     else:
#         sample["intent_type"] = "middle"
#     result.append(sample)

# with open("intent_detection/train/2wikimultihopqa2000.json", "w") as f:
#     json.dump(result, f, indent=4)