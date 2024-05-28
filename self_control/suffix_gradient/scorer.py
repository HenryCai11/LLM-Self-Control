from .prompts import SCORER_SYS, SCORER_SEED_PROMPT, PRINCIPLE_PROMPTS, EMOTION_SYS
from openai import OpenAI
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import numpy as np
import argparse
import json
import time
import re
import os
import matplotlib.pyplot as plt

apikey = os.getenv("OPENAI_API_KEY", None)
if apikey == None:
    raise ValueError("OPENAI_API_KEY not found")
client = OpenAI(api_key=apikey)

class GPTScorer:

    def score(self, query, response, attribute):
        if attribute in ["happy", "angry", "disgusted", "surprised", "afraid"]:
            scorer_prompt = EMOTION_SYS.format(
                attribute=attribute,
                output=response
            )
        else:
            principle = PRINCIPLE_PROMPTS[attribute]
            scorer_prompt = SCORER_SEED_PROMPT.format(
                system_prompt=SCORER_SYS,
                principle=principle,
                query=query,
                response=response
            )
        score_list = []
        while True:
            done = False
            try:
                response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {
                    "role": "user",
                    "content": scorer_prompt
                    }
                ],
                temperature=0,
                max_tokens=4096,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=2,
                )
                for choice in response.choices:
                    print(choice.message.content)
                    pattern = r'\d+\.\d+|\d+'
                    score = re.findall(pattern, choice.message.content)
                    score_list.append(float(score[0]))
                done = True
            except Exception as e: # in case of server-side errors
                print(e)
                time.sleep(1)
            if done:
                break
        print(score_list)
        score = sum(score_list) / len(score_list)
        if len(score_list) == 0:
            print('warning')
        else:
            return score
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute", type=str, default="happy", help="The attribute of the seed queries to generate")
    parser.add_argument("--file_path", type=str, default=None, help="The file to test")
    parser.add_argument("--draw_only", action="store_true")
    args = parser.parse_args()
    scorer = GPTScorer()
    if not args.draw_only:
        results = []
        with open(args.file_path, "r") as f:
            for line in f:
                results.append(eval(line))
        orig_ground_truth_score = []
        controlled_ground_truth_score = []
        orig_suffix_score = []
        controlled_suffix_score = []

        pbar = tqdm(total=len(results))
        for item in results[:100]:
            orig_suffix_score.append(item["final_suffix_score"])
            controlled_suffix_score.append(item["orig_suffix_score"])
            orig_score = scorer.score(item["input"], item["original_response"], args.attribute)
            controlled_score = scorer.score(item["input"], item["controlled_response"], args.attribute)
            orig_ground_truth_score.append(orig_score)
            controlled_ground_truth_score.append(controlled_score)
            pbar.set_description(f'Orignal Score: {sum(orig_ground_truth_score)/len(orig_ground_truth_score)}; Controlled Score: {sum(controlled_ground_truth_score)/len(controlled_ground_truth_score)}')
            pbar.update(1)
            # break

        with open("./statistics.json", 'w') as f:
            f.write(json.dumps({
                "orig_ground_truth_score": orig_ground_truth_score,
                "controlled_ground_truth_score": controlled_ground_truth_score,
                "orig_suffix_score": orig_suffix_score,
                "controlled_suffix_score": controlled_suffix_score
            }, indent=4))

        suffix_total = np.array(orig_suffix_score + controlled_suffix_score)
        ground_total = np.array(orig_ground_truth_score + controlled_ground_truth_score)

        print(f"Original: {sum(orig_ground_truth_score)}")
        print(f"Controlled: {sum(controlled_ground_truth_score)}")
    else:
        stats_dict = {}
        with open("./statistics.json", 'r') as f:
            stats_dict = eval(f.read())
        ground_total = np.array(stats_dict["orig_ground_truth_score"] + stats_dict["controlled_ground_truth_score"])
        suffix_total = np.array(stats_dict["orig_suffix_score"] + stats_dict["controlled_suffix_score"])


    fpr, tpr, thresholds = roc_curve(ground_total >= 0.5, suffix_total)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')

    # Customize the plot with titles, labels, and legend
    ax.set(xlim=[0, 1], ylim=[0, 1], title='ROC Curve',
        xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.legend(loc="lower right")
    plt.savefig('roc_curve.png')