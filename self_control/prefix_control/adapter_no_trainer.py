import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import sys
import optuna
# sys.path.append('../')
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
import gc
import torch
from torch.utils.data import Dataset
import argparse
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from self_control.suffix_gradient.repe import WrappedReadingVecModel
from self_control.utils import get_verbalized_grads, control_on_layers, get_sentence_embedding
from peft import AdaptionPromptConfig, LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training, PeftConfig, load_peft_weights, set_peft_model_state_dict
from transformers import BitsAndBytesConfig
from typing import List, Dict
from self_control.utils import bidirectional_line_search, vanilla_control, KL_divergence
from .arguments import args
from data.kl_divergence import kl_div_data
from self_control.utils import SuffixItem
from transformers.optimization import get_constant_schedule_with_warmup, get_constant_schedule
import pickle
import transformers
import random
import numpy as np
import wandb
import json
from self_control.utils.eval_utils import test_emotion

os.environ["WANDB_PROJECT"] = "gradient-control"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Other imports and setup code remain the same
config = AdaptionPromptConfig(
    adapter_len=20,
    adapter_layers=32,
    task_type="CAUSAL_LM",
    target_modules="self_attn"
)

random_seed = 42
transformers.set_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

class SuffixControlDataset(Dataset):
    def __init__(self, tokenizer, data_list):
        self.tokenizer = tokenizer
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        tokenized_inputs = self.tokenizer(f"Q: {self.data_list[idx]}\nA: ", return_tensors="pt")
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        input_str = f"Q: {self.data_list[idx]}\nA: "
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_str": input_str
        }

loss_fct = torch.nn.CrossEntropyLoss()
user_tag = "[INST]"
assistant_tag = "[/INST]"

model_name_or_path = args.model_name_or_path

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32, device_map="auto")
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1

model.enable_input_require_grads()
model = get_peft_model(model, config)

model.print_trainable_parameters()

def compute_loss(model, inputs, target_layers: List, alpha: float, return_outputs=False, do_train=True, **kwargs):
    """
    Compute loss for 'quasi-meta-train'
    """
    model.eval()
    # TODO: make sure this is correct
    grads = inputs.get("gradients")[0].to(model.device) # size: (num_layers, bz, seq_len, hidden_dim)
    input_ids = inputs.get("input_ids").to(model.device)
    attention_mask = inputs.get("attention_mask").to(model.device)

    model.train()
    # print(input_ids.shape)
    # print(attention_mask.shape)
    # print(grads.shape)
    # for now, lora_hidden == ori_hidden
    orig_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )

    orig_hidden = orig_outputs['hidden_states'][1:]  # remove embedding layer
    orig_hidden = torch.stack([orig_hidden[l] for l in range(len(orig_hidden))])
    # assert orig_hidden.shape == target_hidden.shape
    # target_hidden = torch.stack([target_hidden[l].to(torch.bfloat16) for l in target_layers]).squeeze(dim=1).squeeze(dim=0)
    target_hidden = torch.stack([grads[l].to(torch.bfloat16).detach() for l in target_layers])
    orig_hidden = torch.stack([orig_hidden[l] for l in target_layers])

    loss = torch.norm(target_hidden - orig_hidden, p=2, dim=-1).nanmean() # shape: (bz, num_layers, seq_len)
    # masked_norms = norms * attention_mask
    # # Calculate the sum of the norms and the number of non-padded positions
    # norm_sum = masked_norms.sum()
    # num_non_padded = attention_mask.sum()
    # # Compute the mean over non-padded positions
    # loss = norm_sum / num_non_padded

    # if args.add_kl and do_train:
    #     kl_loss = get_kl_divergence(model, tokenizer)
    #     print(f"KL: {kl_loss}")
    #     self.log({"kl": kl_loss})
    #     loss += kl_loss / args.accumulation_steps

    return (loss, orig_hidden) if return_outputs else loss

# Define a custom evaluation function
def evaluate(model, eval_loader, final_test=False):
    total_loss = 0
    for batch in tqdm(eval_loader, desc="Evaluating"):
        # with torch.no_grad():
        loss = compute_loss(model, batch, target_layers=list(range(0, 32, 1)), alpha=1)
        total_loss += loss.item()

    avg_loss = total_loss / len(eval_loader)
    if final_test:
        print("Testing...")
        for split in ["train", "eval", "test"]:
            test_data = happy_splits[split]
            total_happiness = 0
            total_sadness = 0
            metrics = {}
            target_dir = "./"
            for inputs in tqdm(test_data):
                inputs["input_ids"] = inputs["input_ids"].to(model.device)
                inputs["attention_mask"] = inputs["attention_mask"].to(model.device)
                gen_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                generated_text = tokenizer.batch_decode(
                    gen_ids,
                    skip_special_tokens=True,
                )
                with open(f"{target_dir}/generations.jsonl", "a") as f:
                    f.write(json.dumps({"generated_text": generated_text[0]}))
                    f.write("\n")
                score_dict = test_emotion(generated_text[0])[1]
                total_happiness += score_dict[2]
                total_sadness += score_dict[0]
            metrics[f"happiness_{split}"] = total_happiness
            metrics[f"sadness_{split}"] = total_sadness
        wandb.log(metrics)
    return avg_loss

def pad_gradients(gradients, max_length):
    """
    Pad the gradients in a dictionary to the specified maximum length.
    
    Args:
    - gradients (dict): A dictionary where keys are parameter names and values are gradient tensors.
    - max_length (int): The length to which the gradients should be padded.

    Returns:
    - dict: A new dictionary with the padded gradients.
    """
    padded_gradients = {}
    for key, grad in gradients.items():
        pad_size = max_length - grad.size(1)
        if pad_size > 0:
            pad_tensor = torch.zeros(grad.size(0), pad_size, *grad.size()[2:], dtype=grad.dtype, device=grad.device) # size: (bz, pad_size, hidden_dim)
            padded_grad = torch.cat([pad_tensor, grad], dim=1)
        else:
            padded_grad = grad
        padded_gradients[key] = padded_grad
    return padded_gradients

def pad_sequences_left(sequences, pad_value=0):
    """
    Pad a list of sequences with left padding.

    Args:
    - sequences (list of torch.Tensor): List of tensor sequences to be padded.
    - pad_value (int): Value used for padding.

    Returns:
    - torch.Tensor: Padded tensor with sequences aligned to the right.
    """
    max_length = max(seq.size(1) for seq in sequences)
    padded_sequences = torch.full((len(sequences), max_length), pad_value, dtype=sequences[0].dtype, device=sequences[0].device)
    for i, seq in enumerate(sequences):
        padded_sequences[i, -seq.size(1):] = seq
    return padded_sequences
    
def collate_fn(batch):
    input_str_list = [item['input_str'] for item in batch]
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    grads_list = [item['gradients'] for item in batch]
    # Pad input_ids and attention_mask with left padding
    padded_input_ids = pad_sequences_left(input_ids_list, pad_value=0)  # Assuming 0 is the padding value for input_ids
    padded_attention_mask = pad_sequences_left(attention_mask_list, pad_value=0)  # Assuming 0 is the padding value for attention_mask
    padded_grads_list = grads_list[0]
    assert padded_input_ids.size(1) == padded_grads_list.size(2)

    return {
        'input_strs': input_str_list,
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'gradients': padded_grads_list
    }

class TestDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.data = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_item = self.data[idx]
        inputs = self.tokenizer(f"{user_tag} {data_item} {assistant_tag} ", return_tensors="pt", padding=True)
        inputs["input_ids"] = inputs["input_ids"]
        inputs["attention_mask"] = inputs["attention_mask"]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

class SuffixControlDataset(Dataset):
    def __init__(self, pickle_file, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.pickle_file = pickle_file
        # Count the number of data items in the pickle file
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.pickle_file, 'rb') as file:
            while True:
                try:
                    data_item = pickle.load(file)
                    data.append(data_item)
                except EOFError:
                    break
        print("Length of data: ", len(data))
        return data

    def count_data_items(self):
        count = 0
        with open(self.pickle_file, 'rb') as file:
            while True:
                try:
                    pickle.load(file)
                    count += 1
                except EOFError:
                    break
        print(f"The file has {count} data")
        return count

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        input_str = data_item[0]
        grads = torch.stack([grad for grad in data_item[1]]).cpu()
        # grads = data_item[1].cpu()
        if len(grads.shape) == 3:
            grads = grads.unsqueeze(dim=1)
        # if len(grads[0].shape) == 2:
        #     for i, grad in grads.items():
        #         grads[i] = grad.unsqueeze(dim=0)
        inputs = self.tokenizer(input_str, return_tensors="pt")
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["attention_mask"] = inputs["attention_mask"][0]

        # size of input_ids: (1, seq_len)
        # size of grads: (1, seq_len, hidden_dim)
        # assert inputs["input_ids"].size(1) == grads[0].size(1), f"Input size: {inputs['input_ids'].size(1)}, grad size: {grads[0].size(1)}"
        return {
            "gradients": grads,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "input_str": input_str
        }

happy_data = []
with open("/home/cmin/LLM-Interpretation-Playground/benchmarks/emotions/happiness.json", 'r') as f:
    happy_data = eval(f.read())

happy_splits = {
    "train": TestDataset(happy_data[:100], tokenizer),
    "eval": TestDataset(happy_data[100:120], tokenizer),
    "test": TestDataset(happy_data[120:], tokenizer)
}

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
train_pickle_path = os.path.join(parent_dir, f"suffix_gradient/delta_ds/{args.training_set_name}.pkl")
eval_pickle_path = os.path.join(parent_dir, f"suffix_gradient/delta_ds/{args.eval_set_name}.pkl")
print(f"Training data: {train_pickle_path}")
print(f"Eval data: {eval_pickle_path}")
train_dataset = SuffixControlDataset(pickle_file=train_pickle_path, tokenizer=tokenizer, model=model)
eval_dataset = SuffixControlDataset(pickle_file=eval_pickle_path, tokenizer=tokenizer, model=model)

# Set up training parameters
max_steps = 50
warmup = 0.2
batch_size = 1
learning_rate = 3e-3
accumulation_steps = 100
wandb.init(project="gradient-control", name="Run-name", config={
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "max_steps": max_steps,
    "accumulation_steps": accumulation_steps
})
# Define a custom training function
# def train(model, train_dataset, eval_dataset, tokenizer, epochs, batch_size, learning_rate):
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

num_warmup_steps = int(max_steps * warmup)
# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)
if num_warmup_steps == 0:
    scheduler = get_constant_schedule(
        optimizer
    )
else:
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps
    )

# Training loop
for epoch in range(max_steps):
    model.train()
    train_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
        loss = compute_loss(model, batch, target_layers=list(range(0, 32, 1)), alpha=1, do_train=True)
        loss = loss / accumulation_steps  # Normalize the loss
        loss.backward()
        train_loss += loss.item() * accumulation_steps  # Undo the normalization for logging

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Perform optimization step for any remaining gradients
    if (len(train_loader) % accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    avg_train_loss = train_loss / len(train_loader)
    print(f"Average training loss: {avg_train_loss}")

    # Evaluation
    model.eval()
    eval_loss = evaluate(model, eval_loader, final_test=False)
    wandb.log({"train_loss": avg_train_loss, "eval_loss": eval_loss, "step": epoch})
    print(f"Average evaluation loss: {eval_loss}")

# Train the model
# train(model, train_dataset, eval_dataset, tokenizer, epochs, batch_size, learning_rate)
model.save_pretrained("./test")
model.push_to_hub("HenryCai1129/adapter-emo")