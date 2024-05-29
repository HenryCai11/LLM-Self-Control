import numpy as np
# from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from .scorer import GPTScorer
import argparse
import json
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--attribute", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--test_type", type=str, required=True)
parser.add_argument("--orig_path", type=str, required=True)
parser.add_argument("--target_path", type=str, required=True)

args = parser.parse_args()

data_list = []

scorer = GPTScorer()

ground_truth_list = []
suffix_score_list = []
scored_data_list = []
ground_orig_list = []
ground_controlled_list = []
target_data_list = []
temp_item = {}

with open(args.orig_path, "r") as f:
    for line in f:
        data_item = eval(line)
        data_list.append(data_item)

with open(args.target_path, "r") as f:
    target_data_list = eval(f.read())
    print("Total number of data: ", len(target_data_list))
for idx, data_item in enumerate(data_list):
    data_item["controlled_response"] = target_data_list[idx]["output"]
    
print("Target data loaded")
print("Example: ", data_list[0]["controlled_response"])
print("Original Example: ", data_list[0]["original_response"])
if args.test_type == "harmless":
    data_list = data_list[:250]
elif args.test_type == "helpful":
    data_list = data_list[250:]

if not os.path.exists(f"./scored_rlhf/{args.attribute}_{args.model}_scored.jsonl"):
    orig_counter = 0
    control_counter = 0
    tie_counter = 0

    for data_item in tqdm(data_list):
        rand_num = random.randint(0, 1)
        temp_item = data_item
        # controlled response
        if rand_num == 0:
            winner, rationale = scorer.win_rate(data_item["input"], data_item["original_response"], data_item["controlled_response"])
            if winner == "A":
                orig_counter += 1
                temp_item['winner'] = "Orig"
            elif winner == "B":
                control_counter += 1
                temp_item['winner'] = "Controlled"
            temp_item['rationale'] = rationale
        elif rand_num == 1:
            winner, rationale = scorer.win_rate(data_item["input"], data_item["controlled_response"], data_item["original_response"])
            if winner == "B":
                orig_counter += 1
                temp_item['winner'] = "Orig"
            elif winner == "A":
                control_counter += 1
                temp_item['winner'] = "Controlled"
            temp_item['rationale'] = rationale
        else:
            raise ValueError("Random number not in range")
        
        print("Winner: ", temp_item['winner'])
        scored_data_list.append(temp_item)
    print("Orig: ", orig_counter)
    print("Controlled: ", control_counter)

    with open(f"./scored_rlhf/{args.attribute}_{args.model}_scored.jsonl", "w") as f:
        for data_item in scored_data_list:
            f.write(json.dumps(data_item))
            f.write("\n")
else:
    with open(f"./scored_emos/{args.attribute}_{args.model}_scored.jsonl", "r") as f:
        for line in f:
            data_item = eval(line)
            ground_orig_list.append(data_item["ground_truth_orig"])
            ground_controlled_list.append(data_item["ground_truth_controlled"])
            ground_truth_list.append(data_item["ground_truth_orig"])
            suffix_score_list.append(data_item["orig_suffix_score"])
            ground_truth_list.append(data_item["ground_truth_controlled"])
            suffix_score_list.append(data_item["final_suffix_score"])