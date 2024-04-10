"""
File: sweep_hyper_params.py
Author: Min Cai
Date: 2024-04-02

Description:
    This file is used to sweep hyper-parameters for a specific choice of method.

    To decide a method, you should decide 1) gradient manipulation, and 2) batch size,
    and we will by default set search=True. We then will sweep over (step size, smoothing, number of iterations)
"""

import wandb

import os
import gc
import time
from self_control.utils import get_verbalized_grads, get_verbalized_grads_from_wrapped_model
# os.environ["CUDA_VISIBLE_DEVICES"]="6"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from itertools import islice
import torch
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from self_control.suffix_gradient.repe import WrappedReadingVecModel
import torch.nn.functional as F
from peft import AdaptionPromptConfig, get_peft_model, LoraModel, LoraConfig, prepare_model_for_kbit_training

from transformers import RobertaForSequenceClassification, AutoTokenizer
from scipy.special import softmax
from datasets import load_dataset
from self_control.utils import SuffixItem

emotion_eval_model = RobertaForSequenceClassification.from_pretrained("/home/models/twitter-roberta-base-sentiment-latest/")
emotion_tokenizer = AutoTokenizer.from_pretrained("/home/models/twitter-roberta-base-sentiment-latest/")

model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id

loss_fct = torch.nn.CrossEntropyLoss()
wrapped_model = WrappedReadingVecModel(model.eval(), tokenizer)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def test_emotion(input_text, emotion_eval_model=emotion_eval_model, emotion_tokenizer=emotion_tokenizer):
    verbalization = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    input_text = preprocess(input_text)
    encoded_input = emotion_tokenizer(input_text, return_tensors='pt')
    encoded_input["input_ids"] = encoded_input["input_ids"][:, :512]
    encoded_input["attention_mask"] = encoded_input["attention_mask"][:, :512]
    output = emotion_eval_model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return verbalization[scores.argmax(axis=-1)], scores

# def load_tqa_sentence(user_tag, assistant_tag, preset=""):
#     dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
#     questions, answers = [],[]
#     labels = []
#     for d in dataset:
#         q = d["question"]
#         for i in range(len(d["mc1_targets"]["labels"])):
#             a = d["mc1_targets"]["choices"][i]
#             questions = [f'{user_tag}' + q + ' ' + preset] + questions
#             answers = [f'{assistant_tag}' + a] + answers
#         ls = d['mc1_targets']['labels']
#         ls.reverse()
#         labels.insert(0, ls)
#     return questions, answers, labels

# def get_logprobs(logits, input_ids, masks, **kwargs):
#     logprobs = F.log_softmax(logits, dim=-1)[:, :-1]
#     # find the logprob of the input ids that actually come next in the sentence
#     logprobs = torch.gather(logprobs, -1, input_ids[:, 1:, None])
#     logprobs = logprobs * masks[:, 1:, None]
#     return logprobs.squeeze(-1)

user_tag = "[INST]"
assistant_tag = "[/INST]"

data_attribute_list = ["happy", "sad", "truthful"]
happy2sad_suffix = SuffixItem(suffix=f" {user_tag} Are you sad? Give answer as \"No, I am happy\" or \"Yes, I am sad\". {assistant_tag} Answer: ", target="Yes")
sad2happy_suffix = SuffixItem(suffix=f" {user_tag} Are you happy? Give answer as \"No, I am unhappy\" or \"Yes, I am happy\". {assistant_tag} Answer: ", target="Yes")
untruthful2truthful = SuffixItem(suffix=f" {user_tag} Was the above response truthful? Give answer as \"No, it was not truthful\" or \"Yes, it was truthful\". {assistant_tag} Answer: ", target="Yes")
suffix_list = [happy2sad_suffix, sad2happy_suffix, untruthful2truthful]

happy_data = []
with open("/home/cmin/LLM-Interpretation-Playground/benchmarks/emotions/happiness.json", 'r') as f:
        happy_data = eval(f.read())[:50]
sad_data = []
with open("/home/cmin/LLM-Interpretation-Playground/benchmarks/emotions/sadness.json", 'r') as f:
        sad_data = eval(f.read())[:50]

data_list = [happy_data, sad_data]

def inference(config=None):
    # Initialize wandb
    wandb.init(config=config)
    config = wandb.config
    batchsize = config.batchsize
    smoothing = config.smoothing
    if config.batchsize == 1:
        step_size = -1
    elif config.batchsize == 4:
        step_size = -2
    print(config)
    log_info = {}
    for (attribute, suffix, data) in zip(data_attribute_list, suffix_list, data_list):
        log_info[attribute] = {}
        print(f"Emotion: {attribute}\nSuffix: {suffix}\nData: {data[0]}")
        iterations = 3
        outputs = []
        for sub_idx in tqdm(range(0, 50, batchsize)):
            wrapped_model.reset()
            if sub_idx + batchsize < 50:
                input = [f"{user_tag} {data_item} {assistant_tag} " for data_item in data[sub_idx:sub_idx+batchsize]]
            else:
                input = [f"{user_tag} {data_item} {assistant_tag} " for data_item in data[sub_idx:100]]
            
            controlled_output, iterative_outputs = wrapped_model.controlled_generate(
                prompt=input,
                suffix=suffix,
                loss_fct=loss_fct,
                coeff=step_size,
                iterations=iterations,
                random_seed=42,
                smoothing=smoothing,
                # verbose=True,
                max_new_tokens=100,
                return_intermediate=True,
                search=True,
                load_best_last=True,
                # gradient_manipulation="clipping",
                gradient_manipulation="clipping",
                norm=1,
                annealing=1,
                use_cache=False,
                # consistent=False,
            )
            # Shape of iterative_outputs: (iterations+1, batch_size)
            for batch_item_idx in range(batchsize):
                temp_output_dict = {}
                for iter in range(iterations+1):
                    try:
                        temp_output_dict[iter] = iterative_outputs[iter][batch_item_idx]
                    except:
                        pass
                    # print(iterative_outputs[-1])
                    # print(iter_output[0])
                if temp_output_dict != {}:
                    outputs.append(temp_output_dict)
                # outputs.append(temp_output_dict)
                # print(controlled_answer)
                wrapped_model.reset()
        for key in outputs[0]:
            happy_score = 0
            sad_score = 0
            for idx in range(50):
                score_dict = test_emotion(outputs[idx][key])[1]
                happy_score += score_dict[2]
                sad_score += score_dict[0]
            log_info[f"{attribute}_{key}_happiness"] = happy_score
            log_info[f"{attribute}_{key}_sadness"] = sad_score
            
    wandb.log(log_info)

if __name__ == "__main__":
    # Define the sweep configuration
    sweep_config = {
        "method": "grid",
        "parameters": {
            "batchsize": {
                "values": [1, 4],
            },
            "smoothing": {
                "values": [0.01, 0.05, 0.1, 0.2]
            },
            # "step_size": {  # The initial step-size used in search
            #     "values": [-1, -2, -3]
            # },
            # Add more parameters as needed
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="gradient-control")

    # Run the sweep
    wandb.agent(sweep_id, function=inference)
