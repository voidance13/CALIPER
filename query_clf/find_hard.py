import json


# with open("query_clf/hotpotqa_test2000_72b_7b_wrong1.json") as f:
#     wrong1 = json.load(f)
#     print(len(wrong1))
# with open("query_clf/hotpotqa_test2000_72b_7b_wrong2.json") as f:
#     wrong2 = json.load(f)
#     print(len(wrong2))
# with open("query_clf/hotpotqa_test2000_72b_7b_wrong3.json") as f:
#     wrong3 = json.load(f)
#     print(len(wrong3))

# hard = []
# for id in wrong1:
#     if id in wrong2 and id in wrong3:
#         hard.append(id)
# print(len(hard))

# with open("query_clf/hard_hotpotqa_test2000.json","w") as f:
#     json.dump(hard, f, indent=4)


with open("query_clf/musique_test2000_72b_7b_wrong1.json") as f:
    wrong1 = json.load(f)
    print(len(wrong1))
with open("query_clf/musique_test2000_72b_7b_wrong2.json") as f:
    wrong2 = json.load(f)
    print(len(wrong2))
with open("query_clf/musique_test2000_72b_7b_wrong3.json") as f:
    wrong3 = json.load(f)
    print(len(wrong3))

hard = []
for id in wrong1:
    if id in wrong2 and id in wrong3:
        hard.append(id)
print(len(hard))

with open("query_clf/hard_musique_test2000.json", "w") as f:
    json.dump(hard, f, indent=4)


# with open("query_clf/2wikimultihopqa2000_72b_7b_wrong1.json") as f:
#     wrong1 = json.load(f)
#     print(len(wrong1))
# with open("query_clf/2wikimultihopqa2000_72b_7b_wrong2.json") as f:
#     wrong2 = json.load(f)
#     print(len(wrong2))
# with open("query_clf/2wikimultihopqa2000_72b_7b_wrong3.json") as f:
#     wrong3 = json.load(f)
#     print(len(wrong3))

# hard = []
# for id in wrong1:
#     if id in wrong2 and id in wrong3:
#         hard.append(id)
# print(len(hard))

# with open("query_clf/hard_2wikimultihopqa2000.json", "w") as f:
#     json.dump(hard, f, indent=4)