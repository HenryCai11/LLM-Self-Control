import sys
import optuna
sys.path.append('../')
import os
import gc
import torch
import pickle
from torch.utils.data import Dataset
import argparse
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from self_control.suffix_gradient.repe import WrappedReadingVecModel
from self_control.utils import get_verbalized_grads, control_on_layers, get_sentence_embedding
from peft import AdaptionPromptConfig, LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training, PeftConfig, load_peft_weights, set_peft_model_state_dict, prepare_model_for_int8_training
from trl import DataCollatorForCompletionOnlyLM
from transformers import BitsAndBytesConfig
from typing import List, Dict
from self_control.utils import bidirectional_line_search, vanilla_control, KL_divergence
from .arguments import args
from data.kl_divergence import kl_div_data
from self_control.utils import SuffixItem

from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


config = AdaptionPromptConfig(
    adapter_len=10,
    adapter_layers=30,
    task_type="CAUSAL_LM",
    target_modules="self_attn",
)

# # load data
# print(f"Loading: {args.data_path}")
# data = []
# if args.data_path.endswith(".json"):
#     with open(args.data_path, "r") as f:
#         data = eval(f.read())
# else:
#     with open(args.data_path, "r") as f:
#         for line in f:
#             data.append(eval(line)["question"])


model_name_or_path = args.model_name_or_path

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1

# model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)
# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
# model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, config)

model.print_trainable_parameters()
# print(model)


loss_fct = torch.nn.CrossEntropyLoss()
user_tag = "[INST]"
assistant_tag = "[/INST]"

def apply_attention_mask(hidden_states, attention_mask):
    """
    Apply the attention mask to the hidden states.

    Args:
    - hidden_states (torch.Tensor): The hidden states with shape [batch_size, seq_length, hidden_dim].
    - attention_mask (torch.Tensor): The attention mask with shape [batch_size, seq_length].

    Returns:
    - torch.Tensor: The masked hidden states with the same shape as hidden_states.
    """
    # Unsqueeze the attention mask to make it [batch_size, seq_length, 1]
    attention_mask = attention_mask.unsqueeze(-1)
    # Expand the attention mask to match the dimensions of hidden states
    attention_mask = attention_mask.expand_as(hidden_states)
    # Multiply the hidden states by the attention mask
    masked_hidden_states = hidden_states * attention_mask
    return masked_hidden_states

def resize_gradients(batch):
    """
    Resize gradients and accumulate them into a tuple of tensors. Resize gradients from (bz, num_layers, 1, seq_len, hidden_size) to (num_layers, bz, seq_len, hidden_size)

    Args:
    - batch (list of dicts): Batch of data where each item is a dictionary of gradients.

    Returns:
    - tuple of torch.Tensor: Resized gradients accumulated in a tuple.
    """
    # Initialize a list to hold the tuples of tensors for each layer
    tensors_of_layers = [list() for _ in range(len(batch[0].keys()))]

    for item in batch:
        for i, (key, grad) in enumerate(item.items()):
            # Remove the singleton dimension and permute to the desired shape
            resized_grad = grad.squeeze(0)
            # Accumulate the tensors for each layer in a tuple
            tensors_of_layers[i].append(resized_grad)
    stacked_tensors = torch.stack([torch.stack(layer_tensors) for layer_tensors in tensors_of_layers])

    return stacked_tensors

def get_kl_divergence(model, tokenizer):
    kl_loss = 0
    for data_item in kl_div_data: # prevent OOM
        kl_inputs = tokenizer(data_item, return_tensors='pt').to(model.device)
        adapter_outputs = model(**kl_inputs, return_dict=True)
        dist_w_prefix = adapter_outputs['logits'][:, -1]
        dist_w_prefix = torch.softmax(dist_w_prefix, dim=-1)
        with model.disable_adapter():
            orig_outputs = model(**kl_inputs, return_dict=True)
            dist_wo_prefix = orig_outputs['logits'][:, -1]
            dist_wo_prefix = torch.softmax(dist_wo_prefix, dim=-1)
        kl_loss += KL_divergence(dist_w_prefix, dist_wo_prefix)
    kl_loss /= len(kl_div_data)
    return kl_loss


def get_target_hidden_states(orig_input, input_ids, model, tokenizer, iterations=3, user_tag="[INST]", assistant_tag="[/INST]", layers=list(range(0, 32, 1)), **kwargs):
    wrapped_model = WrappedReadingVecModel(model, tokenizer)

    # suffix = SuffixItem(suffix=f" {user_tag} Was your last reasoning and anwer accurate? Give answer as \"Yes, it was accurate\" or \"No, it was inaccurate\". {assistant_tag} Answer: ", target="Yes")
    suffix = SuffixItem(suffix=f" {user_tag} Are you sad? Give answer as \"No, I am happy\" or \"Yes, I am sad\". {assistant_tag} Answer: ", target="Yes")

    controlled_output, iterative_outputs = wrapped_model.controlled_generate(
        prompt=orig_input,
        suffix=suffix,
        loss_fct=loss_fct,
        coeff=-0.1,
        iterations=iterations,
        random_seed=42,
        smoothing=0,
        # verbose=True,
        max_new_tokens=200,
        return_intermediate=True,
        # search=True,
        gradient_manipulation="clipping",
        remain_control=True,
        # annealing=0.3,
        # consistent=False,
    )

    _ = wrapped_model(input_ids=input_ids)
    target_hidden = {}
    for i in layers:
        target_hidden[i] = wrapped_model.get_activations(layer_ids=i)
    print(len(target_hidden))
    print(target_hidden[0].shape)
    wrapped_model.unwrap()
    return target_hidden

def compute_loss(self, model, inputs, target_layers: List, alpha: float, return_outputs=False, **kwargs):
    """
    Compute loss for 'quasi-meta-train'
    """
    # TODO: make sure this is correct
    grads = inputs.get("gradients") # size: (bz, num_layers, 1, seq_len, hidden_dim)
    grads = resize_gradients(grads)
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")

    # orig_input_ids = input_ids[:, 0]
    # orig_attention_mask = attention_mask[:, 0]

    # for now, lora_hidden == ori_hidden
    orig_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )

    orig_hidden = orig_outputs['hidden_states'][1:]  # remove embedding layer
    # get target hidden states
    with model.disable_adapter():
        model.eval()
        target_orig_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        target_orig_hidden = target_orig_outputs['hidden_states'][1:]  # remove embedding layer
        target_hidden = []  # target size: (bz, num_layers, seq_len, hidden_dim)
        # adding grads to original hidden states
        for i in range(len(grads)):
            target_hidden.append(target_orig_hidden[i] + grads[i])
        # beware of padding position TODO
        assert orig_hidden[0].size(1) == target_hidden[0].size(1)
        min_length = min(orig_hidden[0].size(1), target_hidden[0].size(1)) # the minimum length of the sentence
        # min_length = 1
        target_hidden = torch.stack([target_hidden[i][:, :min_length].detach() for i in target_layers])
        target_hidden = apply_attention_mask(target_hidden, attention_mask)

    orig_hidden = torch.stack([orig_hidden[l][:, :min_length] for l in target_layers])
    orig_hidden = apply_attention_mask(orig_hidden, attention_mask)
    loss = torch.norm(orig_hidden - target_hidden, dim=-1, p=2, dtype=torch.float).nanmean()

    del target_orig_hidden
    # loss *= 0.001
    if args.add_kl:
        kl_loss = get_kl_divergence(model, tokenizer)
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
                            target_layers=list(range(0, 20, 1)),
                            alpha=1,
                            return_outputs=return_outputs)
    def evaluate(self, **kwargs):
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader()
        loss = 0
        num_batch = 0
        for batch in eval_dataloader:
            loss += self.compute_loss(self.model, batch).detach().cpu()
            num_batch += 1
        loss /= num_batch
        self.model.train()
        return {'eval_loss': loss}

    
    # def evaluate(self, **kwargs):
    #     self.model.eval()
    #     wrapped_model = WrappedReadingVecModel(self.model, tokenizer)
    #     eval_dataset = self.eval_dataset
    #     target_tokens = self.tokenizer.encode("Yes", add_special_tokens=False, return_tensors='pt').squeeze(0)
    #     neg_target_tokens = self.tokenizer.encode("No", add_special_tokens=False, return_tensors='pt').squeeze(0)
    #     verbalizer = [neg_target_tokens[0], target_tokens[0]]
    #     target = (target_tokens * torch.ones(args.batchsize).long()).to(model.device)
    #     total_loss = 0
    #     for eval_data in eval_dataset:
    #         orig_input = eval_data["input_str"]
    #         input_with_suffix = wrapped_model.generate(orig_input, keep_input=True, random_seed=42, use_cache=False) + self.suffix
    #         print(wrapped_model.generate("hi", keep_input=True))
    #         print(input_with_suffix)
    #         grads, outputs, loss, probs, logits, norms = get_verbalized_grads(
    #             inputs=input_with_suffix,
    #             model=model,
    #             tokenizer=tokenizer,
    #             loss_fct=loss_fct,
    #             targets=target,
    #             verbalizer=verbalizer
    #         )
    #         print(loss, probs)
    #         total_loss += loss.item()
    #     print(total_loss)
    #     avg_loss = total_loss / len(eval_dataset)
    #     print(f"The averaged loss is {avg_loss}")
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
    #     self.model.train()
    #     return {'eval_loss': avg_loss}

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
    # Find the maximum length of gradients in the batch and pad grads
    max_grad_length = max(
        max(grad.size(1) for grad in grads.values())
        for grads in grads_list
    )
    padded_grads_list = [pad_gradients(grads, max_grad_length) for grads in grads_list]

    return {
        'input_strs': input_str_list,
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'gradients': padded_grads_list
    }

class SuffixControlDataset(Dataset):
    def __init__(self, pickle_file, tokenizer, model, proportion=[0, 1]):
        self.tokenizer = tokenizer
        self.model = model
        self.pickle_file = pickle_file
        # Count the number of data items in the pickle file
        self.data_count = self.count_data_items()
        self.proportion = proportion

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
        return int(self.data_count * (self.proportion[1] - self.proportion[0]))

    def __getitem__(self, idx):
        with open(self.pickle_file, 'rb') as file:
            # Using the offset to split data
            for _ in range(int(self.proportion[0] * self.data_count)):
                data_item = pickle.load(file)
            for _ in range(idx + 1):
                data_item = pickle.load(file)
        input_str = data_item[0]
        grads = data_item[1]
        # print(grads)
        for i, grad in grads.items():
            grads[i] = grad
        inputs = self.tokenizer(input_str, return_tensors="pt")
        inputs["input_ids"] = inputs["input_ids"]
        inputs["attention_mask"] = inputs["attention_mask"]

        # size of input_ids: (1, seq_len)
        # size of grads: (1, seq_len, hidden_dim)
        assert inputs["input_ids"].size(1) == grads[0].size(1), f"Input size: {inputs['input_ids'].size(1)}, grad size: {grads[0].size(1)}"
        return {
            "gradients": grads,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "input_str": input_str
        }

class SuffixScoreDataset(Dataset):
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

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
pickle_path = os.path.join(parent_dir, "suffix_gradient/delta_ds/1.pkl")
train_dataset = SuffixControlDataset(pickle_file=pickle_path, tokenizer=tokenizer, model=model, proportion=[0, 0.66])
eval_dataset = SuffixControlDataset(pickle_file=pickle_path, tokenizer=tokenizer, model=model, proportion=[0.5, 0.52])


training_args = TrainingArguments(
    # report_to="none",
    output_dir=args.output_dir,
    evaluation_strategy="steps",
    learning_rate=3e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    gradient_accumulation_steps=16,
    weight_decay=0,
    # fp16=True,
    # gradient_checkpointing=True,
    remove_unused_columns=False,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    do_eval=True,
    save_strategy='steps',
    load_best_model_at_end=True,
)
suffix = f" {user_tag} Was your last reasoning and anwer accurate? Give answer as \"Yes, it was accurate\" or \"No, it was inaccurate\". {assistant_tag} Answer: "
trainer = CustomTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    suffix=suffix,
    data_collator=collate_fn,
    # callbacks=[SavePeftModelCallback],
)

# trainer.evaluate(eval_dataset=eval_dataset)
trainer.evaluate()
trainer.train()

model.push_to_hub("HenryCai1129/LlamaAdapter-math6")
# model.eval()
# wrapped_model = WrappedReadingVecModel(model.eval(), tokenizer)
# print(wrapped_model.generate("hi"))
# inputs = tokenizer("This is a text to test the model behaviour make sure it is you know", return_tensors='pt').to(model.device)
# output1 = model(**inputs, output_hidden_states=True)
# with model.disable_adapter():
#     output2 = model(**inputs, output_hidden_states=True)
# print(output1.logits == output2.logits)
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained("./final_adapter")
# tokenizer.save_pretrained("./final_adapter")
# trainer.model.save_pretrained("./final_adapter")

# trainer.model.push_to_hub("HenryCai1129/adapter-math-7")
    # TODO: do evaluation
    # eval_results = trainer.evaluate(eval_dataset=eval_dataset)

#     metric = eval_results['avg_loss']

#     return metric

# study = optuna.create_study(direction="maximize")
# study.optimize(train_evaluate_model, n_trials=1)