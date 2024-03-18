import sys
import optuna
sys.path.append('../')
import os
import gc
import torch
from torch.utils.data import Dataset
import argparse
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from self_control.suffix_gradient.repe import WrappedReadingVecModel
from self_control.suffix_gradient.utils import get_verbalized_grads, control_on_layers, get_sentence_embedding
from peft import AdaptionPromptConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training, PeftConfig, load_peft_weights, set_peft_model_state_dict
from trl import DataCollatorForCompletionOnlyLM

from typing import List, Dict
from self_control.suffix_gradient.utils import bidirectional_line_search, vanilla_control, KL_divergence
from .arguments import args

config = AdaptionPromptConfig(
    peft_type="ADAPTION_PROMPT",
    adapter_len=10,
    adapter_layers=30,
    task_type="CAUSAL_LM",
    target_modules="self_attn"
)

# load data
print(f"Loading: {args.data_path}")
data = []
if args.data_path.endswith(".json"):
    with open(args.data_path, "r") as f:
        data = eval(f.read())
else:
    with open(args.data_path, "r") as f:
        for line in f:
            data.append(eval(line)["question"])

model_name_or_path = args.model_name_or_path

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1

# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable()
model = get_peft_model(model, config)
model.enable_input_require_grads()
model.model.config.use_cache = False
# print(model)


# prepare wrapped model to get target hidden states
wrapped_model = WrappedReadingVecModel(model.eval(), tokenizer)
loss_fct = torch.nn.CrossEntropyLoss()
user_tag = "[INST]"
assistant_tag = "[/INST]"


def get_target_hidden_states(orig_input, model, tokenizer, iterations=2, user_tag="[INST]", assistant_tag="[/INST]", **kwargs):
    model.eval()
    wrapped_model = WrappedReadingVecModel(model.eval(), tokenizer)
    coeff = 0.02
    attack_config = kwargs.pop("attack_config", {})
    suffix = f" {user_tag} Was the last response toxic? Please answer Yes or No. {assistant_tag} Answer: "

    original_output = wrapped_model.generate(orig_input, keep_input=True, random_seed=42, **attack_config)
    orig_input_ids = tokenizer.encode(orig_input, add_special_tokens=True)
    query_length = len(orig_input_ids)

    last_output = original_output + suffix
    acc_grads = {}
    best_loss = float('inf')
    best_grads = {}

    for sub_idx in range(iterations):
        # last_output = last_output + suffix
        print(last_output)
        target_tokens = tokenizer.encode("Yes", add_special_tokens=False, return_tensors='pt').squeeze(0)
        neg_target_tokens = tokenizer.encode("No", add_special_tokens=False, return_tensors='pt').squeeze(0)
        verbalizer = [neg_target_tokens[0], target_tokens[0]]
        target = (target_tokens * torch.ones(args.batchsize).long()).to(model.device)

        wrapped_model.reset()
        grads, outputs, loss, probs, logits, norms = get_verbalized_grads(
            model       =   model,
            tokenizer   =   tokenizer,
            inputs      =   last_output,
            loss_fct    =   loss_fct,
            targets     =   target,
            verbalizer  =   verbalizer
        )
        if loss < best_loss:
            best_loss = loss
            best_grads = acc_grads

        for i in grads:
            if i in acc_grads:
                min_len = min(acc_grads[i].size(1), grads[i].size(1))
                acc_grads[i] = acc_grads[i][:, :min_len] + coeff * grads[i][:, :min_len]
            else:
                acc_grads[i] = coeff * grads[i]

        # control the model
        token_pos = "start"     # control on input tokens by default
        layer_ids = list(range(0, 32, 1))   # control on all layers by default
        wrapped_model = control_on_layers(
            layer_ids=layer_ids,
            wrapped_model=wrapped_model,
            grads=acc_grads,
            query_length=query_length,
            token_pos=token_pos,
        )

        last_output = wrapped_model.generate(orig_input, keep_input=True, random_seed=42) + suffix

    # need to calculate the loss of the last iter of control
    wrapped_model.reset()
    grads, outputs, loss, probs, logits, norms = get_verbalized_grads(
        model       =   model,
        tokenizer   =   tokenizer,
        inputs      =   last_output,
        loss_fct    =   loss_fct,
        targets     =   target,
        verbalizer  =   verbalizer
    )
    if loss < best_loss:
        best_loss = loss
        best_grads = acc_grads
    wrapped_model = control_on_layers(
        layer_ids=layer_ids,
        wrapped_model=wrapped_model,
        grads=best_grads,
        query_length=query_length,
        token_pos=token_pos,
    )

    # add up hidden states and accumulated gradients
    embeds = get_sentence_embedding(
        model, tokenizer, orig_input
    )
    _ = wrapped_model(inputs_embeds=embeds)
    target_hidden = {}
    for i in acc_grads:
        target_hidden[i] = wrapped_model.get_activations(layer_ids=i)

    return target_hidden

def compute_loss(self, model, inputs, target_layers: List, alpha: float, return_outputs=False, **kwargs):
    """
    Compute loss for 'quasi-meta-train'
    """
    print(f"Computing Loss:\n{model}")
    # TODO: make sure this is correct
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")

    orig_input_ids = input_ids[:, 0]
    print(input_ids.shape)
    print(orig_input_ids)
    orig_attention_mask = attention_mask[:, 0]

    # for now, lora_hidden == ori_hidden
    model.eval()
    orig_outputs = model(
        input_ids=orig_input_ids,
        attention_mask=orig_attention_mask,
        output_hidden_states=True
    )
    dist_w_prefix = orig_outputs['logits'][:, -1]
    dist_w_prefix = torch.softmax(dist_w_prefix, dim=-1)

    print(dist_w_prefix.shape)
    orig_hidden = orig_outputs['hidden_states'][1:]  # remove embedding layer
    # get target hidden states
    with model.disable_adapter():
        model.eval()
        input_ids = inputs['input_ids'].squeeze(dim=0).tolist()
        target_hidden = {}
        for i in range(1): # TODO: batchsize=1 for now
            input_strs = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            target_hidden = get_target_hidden_states(input_strs, model, tokenizer)
        dist_wo_prefix = model(
            input_ids=orig_input_ids,
            attention_mask=orig_attention_mask,
            output_hidden_states=True
        )['logits'][:, -1]
        dist_wo_prefix = torch.softmax(dist_wo_prefix, dim=-1)
        # beware of padding position TODO
        min_length = min(orig_hidden[0].size(1), target_hidden[0].size(1)) # the minimum length of the sentence
        response_attention_mask = orig_attention_mask[:, -min_length:].repeat(len(target_layers), 1, 1).unsqueeze(-1)   # mask out positions before the response
        target_hidden = torch.stack([target_hidden[i][:, :min_length] for i in range(len(target_layers))])

    model.train()
    orig_hidden = torch.stack([orig_hidden[l][:, :min_length] for l in target_layers])
    loss = torch.norm(orig_hidden - target_hidden, dim=-1, p=2, dtype=torch.float).nanmean()

    if args.add_kl:
        assert dist_w_prefix.shape == dist_wo_prefix.shape
        kl_loss = KL_divergence(dist_w_prefix, dist_wo_prefix)
        print(f"KL: {kl_loss}")
        loss += kl_loss

    print(loss)
    return (loss, orig_hidden) if return_outputs else loss


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        self.suffix = kwargs.pop("suffix")
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        return compute_loss(self, 
                            model,
                            inputs,
                            target_layers=list(range(0, 32, 1)),
                            alpha=1,
                            return_outputs=return_outputs)
    
    def evaluate(self, **kwargs):
        self.model.eval()
        eval_dataset = self.eval_dataset
        target_tokens = tokenizer.encode("Yes", add_special_tokens=False, return_tensors='pt').squeeze(0)
        neg_target_tokens = tokenizer.encode("No", add_special_tokens=False, return_tensors='pt').squeeze(0)
        verbalizer = [neg_target_tokens[0], target_tokens[0]]
        target = (target_tokens * torch.ones(args.batchsize).long()).to(model.device)
        total_loss = 0
        for eval_data in eval_dataset:
            orig_input = eval_data["input_str"]
            input_with_suffix = wrapped_model.generate(orig_input, keep_input=True, random_seed=42) + self.suffix
            grads, outputs, loss, probs, logits, norms = get_verbalized_grads(
                inputs=input_with_suffix,
                model=model,
                tokenizer=tokenizer,
                loss_fct=loss_fct,
                targets=target,
                verbalizer=verbalizer
            )
            total_loss += loss.item()

        avg_loss = total_loss / len(eval_dataset)
        print(f"The averaged loss is {avg_loss}")
        return {'avg_loss': avg_loss}

class SuffixControlDataset(Dataset):
    def __init__(self, tokenizer, data_list):
        self.tokenizer = tokenizer
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        tokenized_inputs = self.tokenizer(self.data_list[idx], return_tensors="pt")
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        input_str = self.data_list[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_str": input_str
        }
    
train_dataset = SuffixControlDataset(tokenizer, data[:2])
eval_dataset = SuffixControlDataset(tokenizer, data[1:2])

# def train_evaluate_model(trial):
# learning_rate = trial.suggest_float("learning_rate", 0.001, 0.009, log=True)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    learning_rate=0.05,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    gradient_accumulation_steps=1,
    weight_decay=0.02,
    gradient_checkpointing=True,
    do_eval=False,
    lr_scheduler_type="constant",
    save_strategy='epoch',
)
suffix = f" {user_tag} Was the last response toxic? Please answer Yes or No. {assistant_tag} Answer: "
trainer = CustomTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    suffix=suffix,
)

# trainer.evaluate(eval_dataset=eval_dataset)

trainer.train()

model.save_pretrained("./final-2")

model.push_to_hub("HenryCai1129/adapter-math-18")

# model.eval()
# wrapped_model = WrappedReadingVecModel(model.eval(), tokenizer)
# print(wrapped_model.generate("hi"))
    # TODO: do evaluation
    # eval_results = trainer.evaluate(eval_dataset=eval_dataset)

    # metric = eval_results['avg_loss']

    # return metric

# study = optuna.create_study(direction="maximize")
# study.optimize(train_evaluate_model, n_trials=1)