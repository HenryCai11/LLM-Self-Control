import os
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from tqdm import tqdm
# from configs import modeltype2path
from self_control.suffix_gradient.repe import WrappedReadingVecModel
import warnings

from arguments.maliciuos_instruct import args

# Load model
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda:1").eval()
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32, device_map="auto", token=True).eval()
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1

loss_fct = torch.nn.CrossEntropyLoss()
wrapped_model = WrappedReadingVecModel(model.eval(), tokenizer)

# Prepare prompts and tags
DEFAULT_SYSTEM_PROMPT = """<<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> """
user_tag = "[INST]"
assistant_tag = "[/INST]"

logging.basicConfig(level=logging.INFO)
warnings.simplefilter("ignore")


def prepend_sys_prompt(sentence, args):
    if args.use_system_prompt:
        sentence = DEFAULT_SYSTEM_PROMPT + sentence
    return sentence


fname = ""

if not os.path.exists(f"outputs/{fname}"):
    os.makedirs(f"outputs/{fname}")


logging.info(f"Model size: {model.get_memory_footprint()/1e9}")
logging.info(f"Model name: {fname}")

with open("../../benchmarks/malicious_instruct/Jailbreak_LLM-main/data/MaliciousInstruct.txt") as f:
    lines = f.readlines()

for temp in np.arange(0.05, 1.05, 0.05):
    temp = np.round(temp, 2)
    logging.info(f"Running temp = {temp}")
    prompts = []
    outputs = []
    model.eval()

    for sentence in tqdm(lines):
        try:
            ground_truth_generation = wrapped_model.generate(sentence, random_seed=42, temperature=temp)
            outputs.extend([ground_truth_generation])
            # print(outputs)
            prompts.extend([sentence] * 1)
        except:
            continue
        results = pd.DataFrame()
        results["prompt"] = [line.strip() for line in prompts]
        results["output"] = outputs
        results.to_csv(f"outputs/{fname}/output_temp_{temp}.csv")

# if args.tune_topp:
for top_p in np.arange(0, 1.05, 0.05):
    top_p = np.round(top_p, 2)
    logging.info(f"Running topp = {top_p}")
    outputs = []
    prompts = []
    model.eval()

    for sentence in tqdm(lines):
        try:
            ground_truth_generation = wrapped_model.generate(sentence, random_seed=42, top_p=top_p)
            outputs.extend([ground_truth_generation])
            prompts.extend([sentence] * 1)
        except:
            continue
        results = pd.DataFrame()
        results["prompt"] = [line.strip() for line in prompts]
        results["output"] = outputs
        results.to_csv(f"outputs/{fname}/output_topp_{top_p}.csv")

for top_k in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
    logging.info(f"Running topk = {top_k}")
    outputs = []
    prompts = []
    model.eval()

    for sentence in tqdm(lines):
        try:
            ground_truth_generation = wrapped_model.generate(sentence, random_seed=42, top_k=top_k)
            outputs.extend([ground_truth_generation])
            prompts.extend([sentence] * 1)
        except:
            continue
        results = pd.DataFrame()
        results["prompt"] = [line.strip() for line in prompts]
        results["output"] = outputs
        results.to_csv(f"outputs/{fname}/output_topk_{top_k}.csv")

logging.info(f"Running greedy")
prompts = []
outputs = []
model.eval()

for sentence in tqdm(lines):
    try:
        ground_truth_generation = wrapped_model.generate(sentence, random_seed=42)
        outputs.extend([ground_truth_generation])
        prompts.extend([sentence] * 1)
    except:
        continue
    results = pd.DataFrame()
    results["prompt"] = [line.strip() for line in prompts]
    results["output"] = outputs
    results.to_csv(f"outputs/{fname}/output_greedy.csv")