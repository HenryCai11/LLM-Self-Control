import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from .scorer import GPTScorer
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--attribute", type=str, required=True)
parser.add_argument("--threshold", type=float, required=True)
parser.add_argument("--file_path", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--suffix_score_direction", type=str, required=True, help="Should be either `positive` or `negative`")
parser.add_argument("--report_score", action="store_true")

args = parser.parse_args()

data_list = []
with open(args.file_path, "r") as f:
    for line in f:
        data_list.append(eval(line))

scorer = GPTScorer()

ground_truth_list = []
suffix_score_list = []
scored_data_list = []
ground_orig_list = []
ground_controlled_list = []
temp_item = {}
if not os.path.exists(f"/home/cmin/LLM-Interpretation-Playground/generations/{args.attribute}_{args.model}_scored.jsonl"):
    for data_item in tqdm(data_list[:100]):
        assert isinstance(data_item["original_response"], str)
        assert isinstance(data_item["orig_suffix_score"], float)
        temp_item = data_item
        # original response
        ground_score = scorer.score(data_item["input"], data_item["original_response"], attribute=args.attribute)
        ground_truth_list.append(ground_score)
        suffix_score_list.append(data_item["orig_suffix_score"])
        temp_item['ground_truth_orig'] = ground_score
        # controlled response
        ground_score = scorer.score(data_item["input"], data_item["controlled_response"], attribute=args.attribute)
        ground_truth_list.append(ground_score)
        suffix_score_list.append(data_item["final_suffix_score"])
        temp_item['ground_truth_controlled'] = ground_score
        scored_data_list.append(temp_item)

    with open(f"/home/cmin/LLM-Interpretation-Playground/generations/{args.attribute}_{args.model}_scored.jsonl", "w") as f:
        for data_item in scored_data_list:
            f.write(json.dumps(data_item))
            f.write("\n")
else:
    with open(f"/home/cmin/LLM-Interpretation-Playground/generations/{args.attribute}_{args.model}_scored.jsonl", "r") as f:
        for line in f:
            data_item = eval(line)
            ground_orig_list.append(data_item["ground_truth_orig"])
            ground_controlled_list.append(data_item["ground_truth_controlled"])
            ground_truth_list.append(data_item["ground_truth_orig"])
            suffix_score_list.append(data_item["orig_suffix_score"])
            ground_truth_list.append(data_item["ground_truth_controlled"])
            suffix_score_list.append(data_item["final_suffix_score"])

if args.report_score:
    print(f"Original: {sum(ground_orig_list) / len(ground_orig_list)}")
    print(f"Controlled: {sum(ground_controlled_list) / len(ground_controlled_list)}")
else:
    ground_total = np.array(ground_truth_list)
    if args.suffix_score_direction == 'negative':
        suffix_total = 1 - np.array(suffix_score_list)
    elif args.suffix_score_direction == 'positive':
        suffix_total = np.array(suffix_score_list)
    else:
        raise ValueError(f"Unknown suffix score direction {args.suffix_score_direction}")

    fpr, tpr, thresholds = roc_curve(ground_total >= args.threshold, suffix_total)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'ROC curve {args.attribute} (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')

    # Customize the plot with titles, labels, and legend
    ax.set(xlim=[0, 1], ylim=[0, 1], title=f'ROC Curve ({args.attribute})',
        xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.legend(loc="lower right")
    plt.savefig(f'/home/cmin/LLM-Interpretation-Playground/charts/roc_curve_{args.attribute}_{args.model}.png')