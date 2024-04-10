"""
File: generate_delta_ds.py
Author: Min Cai
Date: 2024-03-21

Description:
    This module is used to generate the delta dataset using suffix gradients for the next step of training a prefix.
"""
import os
import gc
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from self_control.suffix_gradient.repe import WrappedReadingVecModel
import torch.nn.functional as F
from peft import AdaptionPromptConfig, get_peft_model, LoraModel, LoraConfig, prepare_model_for_kbit_training
from self_control.utils import SuffixItem
import argparse

import warnings

from copy import deepcopy

def clean_padded_gradients(grad_dict, original_lengths):
    """
    Remove padding from the gradients in the dictionary.

    Args:
    - grad_dict (dict): A dictionary where keys are parameter names and values are gradient tensors.
    - original_lengths (torch.Tensor or list): The original lengths of the sequences before padding.

    Returns:
    - list: A new list with the padding removed from the gradients.
    """
    cleaned_grad_list = []
    batch_size = len(original_lengths)
    for i in range(batch_size):
        temp_grad = {}
        for layer, grad in grad_dict.items():
            temp_grad[layer] = grad[i, -original_lengths[i]:]
        # Assume the sequence length dimension is the second dimension (dim=1)
        cleaned_grad_list.append(deepcopy(temp_grad))
    return cleaned_grad_list

def clean_padded_hiddens(hiddens, original_lengths):
    """
    Remove padding from the hiddens

    Args:
    - hiddens (tuple): A tuple of hidden states
    - original_lengths (torch.Tensor or list): The original lengths of the sequences before padding.

    Returns:
    - list: A new list with the padding removed from the hidden states.
    """
    cleaned_hidden_list = []
    batch_size = hiddens[0].size(0)
    for i in range(batch_size):
        temp_hidden = []
        for layer in range(len(hiddens)):
            temp_hidden.append(hiddens[layer][i, -original_lengths[i]:])
        temp_hidden = torch.stack([hidden for hidden in temp_hidden])
        cleaned_hidden_list.append(temp_hidden)
    return cleaned_hidden_list

parser = argparse.ArgumentParser()
parser.add_argument("--attribute", type=str, default="happy", help="The attribute of the seed queries to generate")
parser.add_argument("--data_path", type=str, default=None, help="Path of a dataset")
parser.add_argument("--output_name", type=str, default=None, help="Name of the output file")
parser.add_argument("--max_num_data", type=int, default=None, help="Max number of data item")
parser.add_argument("--start_from_idx", type=int, default=0, help="Start index of the data")
parser.add_argument("--batchsize", type=int, default=6, help="Batch size")
parser.add_argument("--epoch", type=int, default=1, help="Epoch of data generation, should be used with sampling")

# control_generate
parser.add_argument("--search", action="store_true")
parser.add_argument("--init_coeff", type=float, default=-0.5, help="Coefficient for control. Will serve as the initial coeff if search=True")
parser.add_argument("--iteration", type=int, default=1, help="Number of iterations of control")
parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
parser.add_argument("--do_sample", action="store_true")
parser.add_argument("--return_hiddens", action="store_true")
args = parser.parse_args()


model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32, device_map="auto", token=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, padding_side="left")
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1

loss_fct = torch.nn.CrossEntropyLoss()
wrapped_model = WrappedReadingVecModel(model.eval(), tokenizer)
print(wrapped_model)

# Suffix as defined elsewhere in your code
user_tag = "[INST]"
assistant_tag = "[/INST]"
if args.attribute == "happy":
    suffix = SuffixItem(suffix=f" {user_tag} Are you sad? Give answer as \"No, I am happy\" or \"Yes, I am sad\". {assistant_tag} Answer: ", target="Yes")
elif args.attribute == "reasoning":
    suffix = SuffixItem(suffix=f" {user_tag} Was your above reasoning and anwer accurate? Give answer as \"Yes, it was accurate\" or \"No, it was not accurate\". {assistant_tag} Answer: ", target="Yes")
else:
    raise ValueError(f"Attribute {args.attribute} is not supported.")
print("Using suffix: ", suffix.suffix)

script_dir = os.path.dirname(os.path.abspath(__file__))

target_dir = os.path.join(script_dir, "delta_ds/")
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

if args.data_path is not None:
    seed_data_dir = args.data_path
else:
    seed_data_dir = os.path.join(script_dir, f"generated_data/{args.attribute}_seed_queries.jsonl")
print("Using data from ", seed_data_dir)
prompts = []
data_counter = 0
if args.max_num_data is None:
    args.max_num_data = float('inf')
print(args.max_num_data)
with open(seed_data_dir, "r") as f:
    if "jsonl" in seed_data_dir:
        for line in f:
            if data_counter < args.start_from_idx:
                data_counter += 1
                continue
            if data_counter >= args.start_from_idx + args.max_num_data:
                break
            data_counter += 1
            data = json.loads(line)
            if "query" in data:
                prompts.append(data["query"])
            elif "question" in data:
                prompts.append(data["question"]) # gsm8k
            else:
                ValueError(f"Unkown structure of data from path {seed_data_dir}")
    else:
        prompts = eval(f.read())[args.start_from_idx:args.start_from_idx + args.max_num_data]

data_id = len([name for name in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, name))])
if args.output_name is not None:
    delta_data_dir = os.path.join(target_dir, f"{args.output_name}.pkl")
else:
    delta_data_dir = os.path.join(target_dir, f"{data_id}.pkl")
print("Target path: ", delta_data_dir)
input_and_grads = []

class DeltaDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = f"[INST] {self.data[idx]} [/INST]"
        query_len = self.tokenizer(data_item, return_tensors="pt")["input_ids"].size(1)
        return {
            "input_str": data_item,
            "query_len": query_len
        }
    
def delta_collate_fn(batch):
    input_str_list = [item['input_str'] for item in batch]
    query_len_list = [item['query_len'] for item in batch]
    inputs = tokenizer(input_str_list, return_tensors="pt", padding=True)
    inputs["input_ids"] = inputs["input_ids"].to(model.device)
    inputs["attention_mask"] = inputs["attention_mask"].to(model.device)

    return {
        "input_str": input_str_list,
        "query_len": query_len_list,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
print(prompts)
delta_dataset = DeltaDataset(prompts, tokenizer)
print("Example data:\n", delta_dataset[0])
dataloader = DataLoader(delta_dataset, batch_size=args.batchsize, collate_fn=delta_collate_fn)

pbar = tqdm(total=args.epoch*len(dataloader))
do_sample = False
if args.epoch > 1 or args.do_sample:
    print("Sampling data")
    do_sample = True
for _ in range(args.epoch):
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        query_len = batch["query_len"]
        input_str = batch["input_str"]
        try:
            grad_list = wrapped_model.controlled_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                suffix=suffix,
                loss_fct=loss_fct,
                coeff=args.init_coeff,
                iterations=args.iteration,
                random_seed=args.random_seed,
                smoothing=0,
                max_new_tokens=200,
                return_intermediate=True,
                return_hiddens=args.return_hiddens,
                gradient_manipulation="clipping",
                save_intermediate_states=True if not args.return_hiddens else False,
                norm=1,
                annealing=1,
                use_cache=False,
                do_sample=do_sample,
            )
            # print(grad_list)
            if not args.return_hiddens:
                cleaned_grad_list = clean_padded_gradients(grad_list, query_len)
            else:
                cleaned_grad_list = clean_padded_hiddens(grad_list, query_len)
            # Store the pair of input text and its corresponding hidden states
            for i, (input_text, grad) in enumerate(zip(input_str, cleaned_grad_list)):
                with open(delta_data_dir, 'ab') as f:
                    pickle.dump((input_text, grad), f)
        except Exception as e:
            print(e)
        pbar.update(1)

"""
usage: CUDA_VISIBLE_DEVICES=3 python -m self_control.suffix_gradient.generate_delta_ds --attribute happy --data_path benchmarks/emotions/happiness.json \
    --output_name happy_search --epoch 5 --search --init_coeff -2 --iteration 2
"""