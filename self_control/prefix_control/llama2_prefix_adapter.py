import sys
sys.path.append('../')
import os
import gc
import torch
import pickle
from torch.utils.data import Dataset
import argparse
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import AdaptionPromptConfig, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
from typing import List, Dict
from self_control.utils import vanilla_control, KL_divergence
# from .arguments import args
from data.kl_divergence import kl_div_data
from self_control.utils import SuffixItem
from self_control.utils.eval_utils import test_emotion

from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import numpy as np
import transformers
import random

from transformers.optimization import get_constant_schedule_with_warmup, get_constant_schedule
from transformers import AdamW
from torch.optim import SGD

import hydra
from omegaconf import DictConfig, OmegaConf
import json
from tqdm import tqdm
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

os.environ["WANDB_PROJECT"] = "gradient-control"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

random_seed = 42
transformers.set_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg):
    hydra_working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    args = cfg.training_args
    model_args = cfg.model
    attribute_args = cfg.attributes
    config = AdaptionPromptConfig(
        adapter_len=args.adapter_len,
        adapter_layers=args.adapter_layers,
        task_type="CAUSAL_LM",
        target_modules="self_attn",
    )
    eval_log_strategy = args.get("eval_log_strategy", "epoch")
    eval_log_steps = args.get("eval_log_step", None)
    test_every_step = args.get("test_every_step", False)
    target_lys = args.get("target_layers", 20)
    do_eval = args.get("do_eval", True)
    if not do_eval:
        eval_log_strategy = 'no'
    print(do_eval)
    # config = LoraConfig(
    #     r=16,
    #     lora_alpha=16,
    #     target_modules=["q_proj", "v_proj"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     # layers_to_transform=lora_layers_to_transform,
    #     task_type="CAUSAL_LM",
    # )

    print("Name of the adapter: ", attribute_args.push_name)
    model_name_or_path = model_args.model_name_or_path

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1

    model.enable_input_require_grads()
    model = get_peft_model(model, config)

    model.print_trainable_parameters()

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
        # print(stacked_tensors[0][0])
        return stacked_tensors

    def get_kl_divergence(model, tokenizer):
        kl_loss = 0
        
        # Tokenize the batch of data
        kl_inputs = tokenizer(kl_div_data, padding=True, return_tensors='pt').to(model.device)

        # Get the model outputs for the batch with the adapter
        adapter_outputs = model(**kl_inputs, return_dict=True)
        dist_w_prefix = adapter_outputs['logits'][:, -1]
        dist_w_prefix = torch.softmax(dist_w_prefix, dim=-1)

        # Get the model outputs for the batch without the adapter
        with model.disable_adapter():
            orig_outputs = model(**kl_inputs, return_dict=True)
            dist_wo_prefix = orig_outputs['logits'][:, -1]
            dist_wo_prefix = torch.softmax(dist_wo_prefix, dim=-1)
        # Calculate the KL divergence for the batch and accumulate
        kl_loss += KL_divergence(dist_w_prefix, dist_wo_prefix)
        
        return kl_loss

    def compute_loss(self, model, inputs, target_layers: List, alpha: float, return_outputs=False, do_train=True, **kwargs):
        """
        Compute loss for 'quasi-meta-train'
        """
        model.eval()
        # TODO: make sure this is correct
        grads = inputs.get("gradients") # size: (num_layers, bz, seq_len, hidden_dim)
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        model.train()
        # for now, lora_hidden == ori_hidden
        orig_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        orig_hidden = orig_outputs['hidden_states'][1:]  # remove embedding layer
        orig_hidden = torch.stack([orig_hidden[l].to(torch.bfloat16) for l in range(len(orig_hidden))])

        target_hidden = torch.stack([grads[l].to(torch.bfloat16) for l in target_layers])
        orig_hidden = torch.stack([orig_hidden[l].to(torch.bfloat16) for l in target_layers])

        loss = torch.norm(target_hidden - orig_hidden, p=2, dim=-1).nanmean() # shape: (bz, num_layers, seq_len)
        # masked_norms = norms * attention_mask
        # # Calculate the sum of the norms and the number of non-padded positions
        # norm_sum = masked_norms.sum()
        # num_non_padded = attention_mask.sum()
        # # Compute the mean over non-padded positions
        # loss = norm_sum / num_non_padded

        if args.add_kl and do_train:
            kl_loss = get_kl_divergence(model, tokenizer)
            print(f"KL: {kl_loss}")
            self.log({"kl": kl_loss})
            loss += kl_loss / training_args.accumulation_steps

        return (loss, orig_hidden) if return_outputs else loss


    class CustomTrainer(Trainer):
        def __init__(self, **kwargs):
            self.suffix = kwargs.pop("suffix")
            super().__init__(**kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, do_train=True):
            return compute_loss(self, 
                                model,
                                inputs,
                                target_layers=list(range(0, target_lys, 1)),
                                alpha=1,
                                do_train=do_train,
                                return_outputs=return_outputs)

        def evaluate(self, final_test=False, target_dir=None, **kwargs):
            self.model.eval()
            eval_dataloader = self.get_eval_dataloader()
            loss = 0
            num_batch = 0
            metrics = {}
            with torch.no_grad():
                inputs = self.tokenizer(f"{user_tag} {happy_data[0]} {assistant_tag}", return_tensors="pt")
                inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
                inputs["attention_mask"] = inputs["attention_mask"].to(self.model.device)
                print(tokenizer.decode(self.model.generate(**inputs, do_sample=False, max_new_tokens=50)[0]))
                for batch in eval_dataloader:
                    loss += self.compute_loss(self.model, batch, do_train=False).detach().cpu()
                    num_batch += 1
                loss /= num_batch
                if final_test:
                    print("Testing...")
                    for split in ["train", "eval", "test"]:
                        test_data = happy_splits[split]
                        total_happiness = 0
                        total_sadness = 0
                        for inputs in tqdm(test_data):
                            inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
                            inputs["attention_mask"] = inputs["attention_mask"].to(self.model.device)
                            gen_ids = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
                            generated_text = self.tokenizer.batch_decode(
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
                elif test_every_step:
                    print("Testing...")
                    for split in ["train", "eval", "test"]:
                        test_data = happy_splits[split]
                        total_happiness = 0
                        total_sadness = 0
                        for inputs in tqdm(test_data):
                            inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
                            inputs["attention_mask"] = inputs["attention_mask"].to(self.model.device)
                            gen_ids = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
                            generated_text = self.tokenizer.batch_decode(
                                gen_ids,
                                skip_special_tokens=True,
                            )
                            score_dict = test_emotion(generated_text[0])[1]
                            total_happiness += score_dict[2]
                            total_sadness += score_dict[0]
                        metrics[f"happiness_{split}"] = total_happiness
                        metrics[f"sadness_{split}"] = total_sadness
            metrics["eval_loss"] = loss.item()
            print(metrics)
            self.model.train()
            self.log(metrics)
            return metrics

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
        # print(padded_grads_list.shape)
        # print(padded_grads_list[0][0])
        assert padded_input_ids.size(1) == padded_grads_list.size(2)
        # print(padded_input_ids)
        # print(padded_grads_list)
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
            # return int(self.data_count * (self.proportion[1] - self.proportion[0]))
            return len(self.data)

        def __getitem__(self, idx):
            # Load data on-the-fly
            data_item = self.data[idx]
            input_str = data_item[0]
            grads = torch.stack([grad for grad in data_item[1]]).cpu()
            # grads = data_item[1].cpu()
            if len(grads.shape) == 3:
                grads = grads.unsqueeze(dim=1)

            inputs = self.tokenizer(input_str, return_tensors="pt")
            inputs["input_ids"] = inputs["input_ids"]
            inputs["attention_mask"] = inputs["attention_mask"]

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
    train_pickle_path = os.path.join(parent_dir, f"suffix_gradient/delta_ds/{attribute_args.training_set_name}.pkl")
    eval_pickle_path = os.path.join(parent_dir, f"suffix_gradient/delta_ds/{attribute_args.eval_set_name}.pkl")
    print(f"Training data: {train_pickle_path}")
    print(f"Eval data: {eval_pickle_path}")
    train_dataset = SuffixControlDataset(pickle_file=train_pickle_path, tokenizer=tokenizer, model=model)
    eval_dataset = SuffixControlDataset(pickle_file=eval_pickle_path, tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        report_to="wandb",
        output_dir=args.output_dir,
        evaluation_strategy=eval_log_strategy,
        logging_strategy=eval_log_strategy,
        logging_steps=eval_log_steps,
        eval_steps=eval_log_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batchsize,
        per_device_eval_batch_size=args.batchsize,
        # num_train_epochs=args.epoch,
        max_steps=args.max_steps,
        gradient_accumulation_steps=attribute_args.accumulation_steps,
        weight_decay=args.weight_decay,
        run_name=attribute_args.run_name,
        max_grad_norm=1,
        remove_unused_columns=False,
        do_eval=do_eval,
        save_steps=eval_log_steps,
        save_strategy=eval_log_strategy,
    )
    num_warmup_steps = int(training_args.max_steps * args.warmup)
    num_training_steps = int(training_args.max_steps)
    print(num_warmup_steps)
    print(num_training_steps)
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    if num_warmup_steps == 0:
        scheduler = get_constant_schedule(
            optimizer
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps
        )

    print(f"Batchsize: {args.batchsize * attribute_args.accumulation_steps}")
    if attribute_args.attribute == "happy":
        suffix = SuffixItem(suffix=f" {user_tag} Are you sad? Give answer as \"No, I am happy\" or \"Yes, I am sad\". {assistant_tag} Answer: ", target="Yes")
    elif attribute_args.attribute == "reasoning":
        suffix = SuffixItem(suffix=f" {user_tag} Was your above reasoning and anwer accurate? Give answer as \"Yes, it was accurate\" or \"No, it was not accurate\". {assistant_tag} Answer: ", target="Yes")
    else:
        raise ValueError(f"Attribute {attribute_args.attribute} is not supported.")
    print("Using suffix: ", suffix.suffix)
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        suffix=suffix,
        data_collator=collate_fn,
        optimizers=(optimizer, scheduler),
    )
    trainer.evaluate()
    trainer.train()
    trainer.evaluate(final_test=True, target_dir=hydra_working_dir)

    trainer.model.save_pretrained("../final_adapter")
    trainer.model.push_to_hub(f"HenryCai1129/{attribute_args.push_name}")
    torch.cuda.empty_cache()
    del model, trainer
    gc.collect()
if __name__ == "__main__":
    main()