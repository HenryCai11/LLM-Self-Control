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
from peft import AdaptionPromptConfig, LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training, PeftConfig, load_peft_weights, set_peft_model_state_dict
from trl import DataCollatorForCompletionOnlyLM
from transformers import BitsAndBytesConfig
from typing import List, Dict
from self_control.utils import bidirectional_line_search, vanilla_control, KL_divergence
from .arguments import args
from data.kl_divergence import kl_div_data
from self_control.utils import SuffixItem
from self_control.utils import test_emotion

from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import numpy as np
import transformers
import random

from transformers.optimization import get_constant_schedule_with_warmup, get_constant_schedule
from transformers import AdamW
from torch.optim import SGD

model_name_or_path = args.model_name_or_path

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    # layers_to_transform=lora_layers_to_transform,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

model.disable_adapter_layers()
print(type(model))
print(model)
model.enable_adapter_layers()
print(type(model))
print(model)