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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from self_control.suffix_gradient.repe import WrappedReadingVecModel
import torch.nn.functional as F
from peft import AdaptionPromptConfig, get_peft_model, LoraModel, LoraConfig, prepare_model_for_kbit_training
from self_control.utils import SuffixItem
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--attribute", type=str, default="happy", help="The attribute of the seed queries to generate")
parser.add_argument("--data_path", type=str, default=None, help="Path of a dataset")
parser.add_argument("--output_name", type=str, default=None, help="Name of the output file")
parser.add_argument("--max_num_data", type=int, default=1000, help="Max number of data item")
args = parser.parse_args()


# model_name_or_path = "/home/models/llama2-7b-chat-hf/"
model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto").eval()
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_d|type=torch.float32, device_map="auto", token=True).eval()
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
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
with open(seed_data_dir, "r") as f:
    for line in f:
        if data_counter >= args.max_num_data:
            break
        data_counter += 1
        data = json.loads(line)
        if "query" in data:
            prompts.append(data["query"])
        elif "question" in data:
            prompts.append(data["question"]) # gsm8k
        else:
            ValueError(f"Unkown structure of data from path {seed_data_dir}")

data_id = len([name for name in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, name))])
if args.output_name is not None:
    delta_data_dir = os.path.join(target_dir, f"{args.output_name}.pkl")
else:
    delta_data_dir = os.path.join(target_dir, f"{data_id}.pkl")
print("Target path: ", delta_data_dir)
input_and_grads = []

for input_text in prompts:
    input_text = f"[INST] {input_text} [/INST]"
    controlled_output, grad_list = wrapped_model.controlled_generate(
        prompt=input_text,
        suffix=suffix,
        loss_fct=loss_fct,
        coeff=-0.4,
        iterations=1,
        random_seed=42,
        smoothing=0,
        max_new_tokens=200,
        return_intermediate=True,
        gradient_manipulation="clipping",
        save_intermediate_states=True,
        norm=1,
        annealing=1,
        use_cache=False,
    )
    # Store the pair of input text and its corresponding hidden states
    for i, grad in enumerate(grad_list):
        with open(delta_data_dir, 'ab') as f:
            pickle.dump((input_text, grad), f)
# Now you have a list where each element is a tuple (input_text, hidden_states)

# Save the data
# You can choose the format that best suits your needs, for example, pickle




# To load the hidden states and inputs later
with open(delta_data_dir, 'rb') as f:
    loaded_data = pickle.load(f)
