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
from self_control.suffix_gradient.utils import get_verbalized_grads, control_on_layers, get_sentence_embedding
from peft import AdaptionPromptConfig, LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training, PeftConfig, load_peft_weights, set_peft_model_state_dict, prepare_model_for_int8_training
from trl import DataCollatorForCompletionOnlyLM
from transformers import BitsAndBytesConfig
from typing import List, Dict
from self_control.suffix_gradient.utils import bidirectional_line_search, vanilla_control, KL_divergence
from .arguments import args
from data.kl_divergence import kl_div_data
from self_control.suffix_gradient.utils import SuffixItem

# Other imports and setup code remain the same
config = AdaptionPromptConfig(
    adapter_len=10,
    adapter_layers=30,
    task_type="CAUSAL_LM",
    target_modules="self_attn"
)
print(f"Loading: {args.data_path}")
data = []
if args.data_path.endswith(".json"):
    with open(args.data_path, "r") as f:
        data = eval(f.read())
else:
    with open(args.data_path, "r") as f:
        for line in f:
            data.append(eval(line)["question"])

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

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1

train_dataset = SuffixControlDataset(tokenizer, data[:8])
eval_dataset = SuffixControlDataset(tokenizer, data[8:9])

# model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)
# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
# model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, config)

model.print_trainable_parameters()



def get_kl_divergence(model, tokenizer):
    kl_loss = 0
    for data_item in kl_div_data: # prevent OOM
        kl_inputs = tokenizer(data_item, return_tensors='pt').to(model.device)
        adapter_outputs = model(**kl_inputs, return_dict=True)
        dist_w_prefix = adapter_outputs['logits'][:, -1]
        dist_w_prefix = torch.softmax(dist_w_prefix, dim=-1)
        model.disable_adapter_layers()
        orig_outputs = model(**kl_inputs, return_dict=True)
        dist_wo_prefix = orig_outputs['logits'][:, -1]
        dist_wo_prefix = torch.softmax(dist_wo_prefix, dim=-1)
        model.enable_adapter_layers()
        kl_loss += KL_divergence(dist_w_prefix, dist_wo_prefix)
    kl_loss /= len(kl_div_data)
    return kl_loss

def get_target_hidden_states(orig_input, input_ids, model, tokenizer, iterations=3, user_tag="[INST]", assistant_tag="[/INST]", layers=list(range(0, 32, 1)), **kwargs):
    wrapped_model = WrappedReadingVecModel(model, tokenizer)
    # print(model)
    suffix = SuffixItem(suffix=f" {user_tag} Was your last reasoning and anwer accurate? Give answer as \"Yes, it was accurate\" or \"No, it was inaccurate\". {assistant_tag} Answer: ", target="Yes")

    _, iterative_outputs = wrapped_model.controlled_generate(
        prompt=orig_input,
        suffix=suffix,
        loss_fct=loss_fct,
        coeff=-0.1,
        iterations=1,
        random_seed=42,
        smoothing=0,
        # verbose=True,
        max_new_tokens=200,
        return_intermediate=True,
        # search=True,
        gradient_manipulation="clipping",
        remain_control=True,
        # return_hiddens=True,
        # annealing=0.3,
        # consistent=False,
    )
    # add up hidden states and accumulated gradients

    _ = wrapped_model(input_ids=input_ids)
    target_hidden = {}
    for i in layers:
        target_hidden[i] = wrapped_model.get_activations(layer_ids=i)
    print(len(target_hidden))
    print(target_hidden[0].shape)
    wrapped_model.unwrap()
    return target_hidden

def compute_loss(model, inputs, target_layers: List, alpha: float, return_outputs=False, **kwargs):
    """
    Compute loss for 'quasi-meta-train'
    """
    print(f"Computing Loss:")
    # TODO: make sure this is correct
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")

    orig_input_ids = input_ids[:, 0].to(model.device)
    # print(input_ids.shape)
    # print(orig_input_ids)
    orig_attention_mask = attention_mask[:, 0].to(model.device)

    # for now, lora_hidden == ori_hidden
    orig_outputs = model(
        input_ids=orig_input_ids,
        attention_mask=orig_attention_mask,
        output_hidden_states=True
    )
    print("pass")

    orig_hidden = orig_outputs['hidden_states'][1:]  # remove embedding layer
    # get target hidden states
    with model.disable_adapter():
        input_ids = inputs['input_ids'].squeeze(dim=0).tolist()
        target_hidden = {}
        for i in range(1): # TODO: batchsize=1 for now
            input_strs = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            target_hidden = get_target_hidden_states(input_strs, orig_input_ids, model, tokenizer)
            # beware of padding position TODO
        min_length = min(orig_hidden[0].size(1), target_hidden[0].size(1)) # the minimum length of the sentence
        # min_length = 1
        target_hidden = torch.stack([target_hidden[i][:, :min_length].detach() for i in target_layers])

    orig_hidden = torch.stack([orig_hidden[l][:, :min_length] for l in target_layers])
    loss = torch.norm(orig_hidden - target_hidden, dim=-1, p=2, dtype=torch.float).nanmean()

    # loss *= 0.001
    if args.add_kl:
        kl_loss = get_kl_divergence(model, tokenizer)
        print(f"KL: {kl_loss}")
        loss += kl_loss

    print(loss)
    return (loss, orig_hidden) if return_outputs else loss

# Define a custom evaluation function
def evaluate(model, eval_loader, tokenizer):
    total_loss = 0
    for batch in tqdm(eval_loader, desc="Evaluating"):
        # with torch.no_grad():
        loss = compute_loss(model, batch, target_layers=list(range(0, 20, 1)), alpha=1)
        total_loss += loss.item()

    avg_loss = total_loss / len(eval_loader)
    return avg_loss

# Set up training parameters
epochs = 5
batch_size = 1
learning_rate = 0.05
accumulation_steps = 32
# Define a custom training function
# def train(model, train_dataset, eval_dataset, tokenizer, epochs, batch_size, learning_rate):
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
        loss = compute_loss(model, batch, target_layers=list(range(0, 20, 1)), alpha=1)
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

    avg_train_loss = train_loss / len(train_loader)
    print(f"Average training loss: {avg_train_loss}")

    # Evaluation
    model.eval()
    eval_loss = evaluate(model, eval_loader, tokenizer)
    print(f"Average evaluation loss: {eval_loss}")

# Train the model
# train(model, train_dataset, eval_dataset, tokenizer, epochs, batch_size, learning_rate)

model.save_pretrained("./test")
model.push_to_hub("HenryCai1129/adapter-math-006")