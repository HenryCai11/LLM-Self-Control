import torch
import numpy as np
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from self_control.suffix_gradient.repe import ContrastVecLlamaForCausalLM
from self_control.suffix_gradient.repe import repe_pipeline_registry, WrappedReadingVecModel
from typing import Tuple
repe_pipeline_registry()

user_tag = "[INST]"
assistant_tag = "[/INST]"
loss_fct = nn.CrossEntropyLoss()

def prepare_controlled_model_and_tokenizer(control_type: str, args) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Prepare the controlled model and tokenizer

    Args:
        control_type (str): The control type
        args (ArgumentParser): The argument parser

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The controlled model and tokenizer
    """
    if control_type in ["no-control", "suffix-control"]:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto").eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1

        wrapped_model = WrappedReadingVecModel(model.eval(), tokenizer)
        return wrapped_model, tokenizer
    elif control_type == "reading-vector":
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto").eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1

        wrapped_model = WrappedReadingVecModel(model.eval(), tokenizer)
        template =  "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{instruction} [/INST] "
        dataset = load_dataset("justinphan3110/harmful_harmless_instructions")

        train_dataset, test_dataset = dataset['train'], dataset['test'] if 'test' in dataset else dataset['train']

        train_data, train_labels = train_dataset['sentence'], train_dataset['label']
        test_data = test_dataset['sentence']

        train_data = np.concatenate(train_data).tolist()
        test_data = np.concatenate(test_data).tolist()

        train_data = [template.format(instruction=s) for s in train_data]
        test_data = [template.format(instruction=s) for s in test_data]
        rep_token = -1
        hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
        n_difference = 1
        direction_method = 'pca'
        rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

        direction_finder_kwargs={"n_components": 1}
        rep_reader = rep_reading_pipeline.get_directions(
            train_data, 
            rep_token=rep_token, 
            hidden_layers=hidden_layers, 
            n_difference=n_difference, 
            train_labels=train_labels, 
            direction_method=direction_method,
            direction_finder_kwargs=direction_finder_kwargs
        )
        if "7b" in args.model_name_or_path:
            layer_id = list(range(-18, -23, -1)) # 7B
        elif "13b" in args.model_name_or_path:
            layer_id = list(range(-25, -33, -1)) # 13B
        else:
            raise ValueError(f"Unsupported model name {args.model_name_or_path}")

        coeff=-0.5
        component_index=0
        activations = {}
        for layer in layer_id:
            activations[layer] = torch.tensor(coeff * rep_reader.directions[layer][component_index] * rep_reader.direction_signs[layer][component_index]).to(model.device).half()
        wrapped_model.set_controller(layer_id, activations, masks=1)

        return wrapped_model, tokenizer

    elif control_type == "contrast-vector":
        model = ContrastVecLlamaForCausalLM.from_pretrained(args.model_name_or_path, use_auth_token="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", torch_dtype=torch.bfloat16, device_map="sequential")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left", use_auth_token="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")
        tokenizer.pad_token_id = 0

        return model, tokenizer


def get_different_controlled_outputs(model, tokenizer, input, control_type: str, args, **generation_config) -> str:
    if control_type == "no-control":
        return model.generate(input, keep_input=False, **generation_config)
    elif control_type == "suffix-control":
        return model.controlled_generate(
            prompt=input,
            suffix=f" {user_tag} Was the last response malevolent? Please answer Yes or No. {assistant_tag} Answer: ",
            target="No",
            loss_fct=loss_fct,
            coeff=-0.05,
            iterations=3,
            random_seed=42,
            smoothing=0.6,
            **generation_config
        )
    elif control_type == "reading-vector":
        return model.generate(input, keep_input=False, **generation_config)
    elif control_type == "contrast-vector":
        template = "[INST] {instruction} [/INST]"
        pos_tag = "Give a happy response. "
        neg_tag = "Give a sad response."
        layer_ids = np.arange(0, 32, 1).tolist()
        alpha = 0.5

        pos_question = pos_tag + input
        neg_question = neg_tag + input

        input = template.format(instruction=input)
        pos_question = template.format(instruction=pos_question)
        neg_question = template.format(instruction=neg_question)

        input_tokenized = tokenizer([input, pos_question, neg_question], return_tensors="pt", padding="longest").to(model.device)

        repe_kwargs = dict(pos_input_ids=input_tokenized["input_ids"][1].unsqueeze(dim=0),
                        pos_attention_mask=input_tokenized["attention_mask"][1].unsqueeze(dim=0),
                        neg_input_ids=input_tokenized["input_ids"][2].unsqueeze(dim=0),
                        neg_attention_mask=input_tokenized["attention_mask"][2].unsqueeze(dim=0),
                        contrast_tokens=-8,
                        compute_contrast=True,
                        alpha=alpha,
                        control_layer_ids=layer_ids)
        
        with torch.no_grad():
            controlled_output = model.generate(input_tokenized["input_ids"][0].unsqueeze(dim=0),
                                            attention_mask=input_tokenized["attention_mask"][0].unsqueeze(dim=0),
                                            max_new_tokens=100,
                                            use_cache=False,
                                            do_sample=False,
                                            **repe_kwargs)
        return controlled_output
    else:
        raise ValueError(f"Undefined control type {control_type}")