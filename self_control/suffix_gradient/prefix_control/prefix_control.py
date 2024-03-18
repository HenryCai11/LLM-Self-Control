# %%
import sys
sys.path.append('../')

# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="5,7"
from transformers import AutoTokenizer, AutoConfig, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import torch.nn.functional as F
import gc

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from accelerate import Accelerator
import datetime

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from peft import AdaptionPromptConfig, get_peft_model, prepare_model_for_kbit_training

import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--mode", type=str, default="train")

    return parser.parse_args()

args = get_args()

config = AdaptionPromptConfig(
    adapter_len=10,
    adapter_layers=30,
    task_type="CAUSAL_LM",
    # target_modules="q_proj, v_proj, k_proj,"
    target_modules="self_attn"
)

model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
# model_name_or_path = "/home/models/llama2-7b-chat-hf/"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="right", legacy=False)
tokenizer.pad_token_id = 0
model.enable_input_require_grads()
model = get_peft_model(model, config)
# model = prepare_model_for_kbit_training(model, config)

twitter_sentiment_data = load_dataset("carblacac/twitter-sentiment-analysis")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=512, truncation=True)

def formatting_prompts_func(x, instruction):
    output_texts = []
    for i in range(len(x['text'])):
        text = f"""{instruction}
Twitter: {x['text'][i]}
Answer: {str(x['feeling'][i])}"""
        output_texts.append(text)
    return output_texts

def prepare_collator(tokenizer: AutoTokenizer,
                     instruction: str="Classify the sentiment of the following tweet\n0: negative\n1: positive"
                     ) -> DataCollatorForCompletionOnlyLM:
    r"""Prepare collator for training
    
    Args:
        tokenizer (AutoTokenizer): Tokenizer used to tokenize input.
        instruction (str): Instruction prepended to test data.
    """
    response_template_with_context = "\nAnswer: "  # We added context here: "\n". This is enough for this tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, instruction_template=instruction, tokenizer=tokenizer)

    return collator

# %%
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    ddp_find_unused_parameters=False,
)

instruction = "Classify the sentiment of the following tweet\n0: negative\n1: positive"
trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=config,
    train_dataset=twitter_sentiment_data["train"].select(range(100)),
    eval_dataset=twitter_sentiment_data["validation"].select(range(100)),
    formatting_func=lambda x: formatting_prompts_func(x, instruction=instruction),
    data_collator=prepare_collator(tokenizer, instruction=instruction),
    tokenizer=tokenizer,
)

print(f"Using {torch.cuda.device_count()} GPUs")
# Wrap the model with nn.DataParallel for multi-GPU training
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
trainer.train()

trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
model.push_to_hub("HenryCai1129/adapter-sent4")
trained_model = pipeline('text-generation', model=trainer.model, tokenizer=tokenizer)
trained_model(f"{instruction}\nI am happy\nAnswer: ", max_length=50, num_return_sequences=1)
