import json


# with open("intent_detection/hotpotqa_test2000_72b_7b_wrong1.json") as f:
#     wrong1 = json.load(f)
#     print(len(wrong1))
# with open("intent_detection/hotpotqa_test2000_72b_7b_wrong2.json") as f:
#     wrong2 = json.load(f)
#     print(len(wrong2))
# with open("intent_detection/hotpotqa_test2000_72b_7b_wrong3.json") as f:
#     wrong3 = json.load(f)
#     print(len(wrong3))

# hard = []
# for id in wrong1:
#     if id in wrong2 and id in wrong3:
#         hard.append(id)
# print(len(hard))

# with open("intent_detection/hard_hotpotqa_test2000.json","w") as f:
#     json.dump(hard, f, indent=4)


with open("intent_detection/musique_test2000_72b_7b_wrong1.json") as f:
    wrong1 = json.load(f)
    print(len(wrong1))
with open("intent_detection/musique_test2000_72b_7b_wrong2.json") as f:
    wrong2 = json.load(f)
    print(len(wrong2))
with open("intent_detection/musique_test2000_72b_7b_wrong3.json") as f:
    wrong3 = json.load(f)
    print(len(wrong3))

hard = []
for id in wrong1:
    if id in wrong2 and id in wrong3:
        hard.append(id)
print(len(hard))

with open("intent_detection/hard_musique_test2000.json", "w") as f:
    json.dump(hard, f, indent=4)


# with open("intent_detection/2wikimultihopqa2000_72b_7b_wrong1.json") as f:
#     wrong1 = json.load(f)
#     print(len(wrong1))
# with open("intent_detection/2wikimultihopqa2000_72b_7b_wrong2.json") as f:
#     wrong2 = json.load(f)
#     print(len(wrong2))
# with open("intent_detection/2wikimultihopqa2000_72b_7b_wrong3.json") as f:
#     wrong3 = json.load(f)
#     print(len(wrong3))

# hard = []
# for id in wrong1:
#     if id in wrong2 and id in wrong3:
#         hard.append(id)
# print(len(hard))

# with open("intent_detection/hard_2wikimultihopqa2000.json", "w") as f:
#     json.dump(hard, f, indent=4)