import json


# with open("intent_detection/hotpotqa_test2000_7b_right1.json") as f:
#     right1 = json.load(f)
#     print(len(right1))
# with open("intent_detection/hotpotqa_test2000_7b_right2.json") as f:
#     right2 = json.load(f)
#     print(len(right2))
# with open("intent_detection/hotpotqa_test2000_7b_right3.json") as f:
#     right3 = json.load(f)
#     print(len(right3))

# easy = []
# for id in right1:
#     if id in right2 and id in right3:
#         easy.append(id)
# print(len(easy))

# with open("intent_detection/easy_hotpotqa_test2000.json","w") as f:
#     json.dump(easy, f, indent=4)


with open("intent_detection/musique_test2000_7b_right1.json") as f:
    right1 = json.load(f)
    print(len(right1))
with open("intent_detection/musique_test2000_7b_right2.json") as f:
    right2 = json.load(f)
    print(len(right2))
with open("intent_detection/musique_test2000_7b_right3.json") as f:
    right3 = json.load(f)
    print(len(right3))

easy = []
for id in right1:
    if id in right2 and id in right3:
        easy.append(id)
print(len(easy))

with open("intent_detection/easy_musique_test2000.json", "w") as f:
    json.dump(easy, f, indent=4)


# with open("intent_detection/2wikimultihopqa2000_7b_right1.json") as f:
#     right1 = json.load(f)
#     print(len(right1))
# with open("intent_detection/2wikimultihopqa2000_7b_right2.json") as f:
#     right2 = json.load(f)
#     print(len(right2))
# with open("intent_detection/2wikimultihopqa2000_7b_right3.json") as f:
#     right3 = json.load(f)
#     print(len(right3))

# easy = []
# for id in right1:
#     if id in right2 and id in right3:
#         easy.append(id)
# print(len(easy))

# with open("intent_detection/easy_2wikimultihopqa2000.json", "w") as f:
#     json.dump(easy, f, indent=4)