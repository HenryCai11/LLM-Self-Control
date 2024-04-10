import pickle
import transformers
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from self_control.prefix_control.arguments import args
from self_control.suffix_gradient.repe import WrappedReadingVecModel
from self_control.utils.eval_utils import test_emotion
from tqdm import tqdm
import json

model_name_or_path = args.model_name_or_path

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1

loss_fct = torch.nn.CrossEntropyLoss()
wrapped_model = WrappedReadingVecModel(model.eval(), tokenizer)

user_tag = "[INST]"
assistant_tag = "[/INST]"

class TestDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.data = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_item = self.data[idx]
        inputs = self.tokenizer(f"{user_tag} {data_item} {assistant_tag}", return_tensors="pt", padding=True)
        inputs["input_ids"] = inputs["input_ids"]
        inputs["attention_mask"] = inputs["attention_mask"]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

random_seed = 42
transformers.set_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

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
        # if len(grads[0].shape) == 2:
        #     for i, grad in grads.items():
        #         grads[i] = grad.unsqueeze(dim=0)
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
    "eval": TestDataset(happy_data[100:110], tokenizer),
    "test": TestDataset(happy_data[110:], tokenizer)
}
train_dataset = SuffixControlDataset(pickle_file="/home/cmin/LLM-Interpretation-Playground/self_control/suffix_gradient/delta_ds/happy_search_100_greedy.pkl", tokenizer=tokenizer, model=model)
gradient_splits = {
    "train": train_dataset,
}
# for split in ["train", "eval", "test"]:
filter_list = []
for split in ["train"]:
    test_data = happy_splits[split]
    gradient_data = gradient_splits[split]
    assert len(gradient_data) == len(test_data)
    total_happiness = 0
    total_sadness = 0
    pbar = tqdm(total=len(test_data), desc='Test emotion')
    model.eval()
    for inputs, grad_data in zip(test_data, gradient_data):
        wrapped_model.reset()
        assert inputs["input_ids"].size(1) == grad_data["input_ids"].size(1)
        inputs["input_ids"] = inputs["input_ids"].to(model.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(model.device)
        hiddens = model(
            **inputs,
            output_hidden_states=True
        )["hidden_states"][1:]
        # print(hiddens[0].shape)
        grads = {}
        for layer in range(0, 1, 1):
            grads[layer] = grad_data["gradients"][layer].to(model.device).detach() - hiddens[layer]
        # print(hiddens[-1])
        # print(grad_data["gradients"][-1])
        wrapped_model.control_on_layers(
            layer_ids=list(range(0, 1, 1)),
            grads=grads,
            query_length=len(grad_data["input_ids"][0]),
            token_pos="start",
            gradient_manipulation="clipping",
            epsilon=1
        )
        gen_ids = wrapped_model.generate(**inputs, max_new_tokens=100, return_ids=True, do_sample=False)
        generated_text = tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True,
        )
        with open(f"./generations.jsonl", "a") as f:
            f.write(json.dumps({"generated_text": generated_text[0]}))
            f.write("\n")
        score_dict = test_emotion(generated_text[0])[1]
        if score_dict[2] < 0.5:
            print(grad_data["input_str"])
            filter_list.append(grad_data["input_str"])
        total_happiness += score_dict[2]
        total_sadness += score_dict[0]
        pbar.set_description(f'Current Happiness: {total_happiness} Current Sadness: {total_sadness}')
        pbar.update(1)
print(filter_list)