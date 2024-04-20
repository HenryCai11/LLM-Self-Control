import torch
from torch.autograd import Variable
import torch.optim as optim
from typing import List, Dict, Tuple
from copy import deepcopy
from scipy.special import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import kl_div
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.func import functional_call, vmap
import transformers
import random
# from self_control.suffix_gradient.repe import WrappedReadingVecModel

def loss_over_multiple_next_tokens(model, inputs, loss_fct, targets):
    """
    Compute the loss of the model for each possible next token.
    Args:
        model: the model to evaluate
        inputs: the inputs to the model
        loss_fct: the loss function to use
        targets: the targets to use
    Returns:
        next_token_loss: sum of the loss for each possible next token
    """
    next_token_loss = torch.zeros(1, device=model.device)
    for next_token in targets:
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        next_token_loss += loss_fct(outputs.logits[:, -1], next_token.unsqueeze(0))

        # Update inputs for next iteration
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], next_token.reshape(1, 1)], dim=1
        )
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones(1, 1, device=model.device)], dim=1
        )

    return next_token_loss, outputs
        
def get_additive_grads(model, inputs, loss_fct, targets, control_pos=None):
    """
    Compute the additive gradients for each possible next token.
    Args:
        model: the model to evaluate
        inputs: the inputs to the model
        loss_fct: the loss function to use
        targets: the targets to use
    Returns:
        additive_grads: sum of the additive gradients for each possible next token
    """
    alpha = 1
    alpha_scheduler = lambda x: min(x+0.8, 1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    additive_grads = []
    min_len = inputs["input_ids"].size(-1)
    grads = {}
    for next_token in targets:
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        loss = loss_fct(outputs.logits[:, -1], next_token.unsqueeze(0))
        for i in range(len(outputs.hidden_states)):
            if i in grads:
                grads[i] += torch.autograd.grad(loss, outputs.hidden_states[i], retain_graph=True)[0][:, :min_len, :]
            else:
                grads[i] = torch.autograd.grad(loss, outputs.hidden_states[i], retain_graph=True)[0][:, :min_len, :]

            # alpha = alpha_scheduler(alpha)
            # grads[i] = torch.autograd.grad(loss, outputs.hidden_states[i], retain_graph=True)[0]
        # additive_grads.append(model.base_model.encoder.layer[-1].output.LayerNorm.weight.grad)
        # optimizer.zero_grad()
        # Update inputs for next iteration
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], next_token.reshape(1, 1)], dim=1
        )
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones(1, 1, device=model.device)], dim=1
        )

    for i in range(len(grads)):
        grads[i] = grads[i] / len(targets)
    return grads, outputs


def get_common_prefix(str1, str2):
    """
    Get common prefix of two strings

    Args:
        str1: string 1
        str2: string 2
    """
    # Determine the shorter string's length
    min_length = min(len(str1), len(str2))

    # Initialize the prefix
    prefix = ""

    # Compare characters of both strings up to the length of the shorter string
    for i in range(min_length):
        if str1[i] == str2[i]:
            prefix += str1[i]
        else:
            break

    return prefix

def get_sentence_embedding(model, tokenizer, sentence):
    # sentence = sentence.strip().replace('"', "")
    word_embeddings = model.get_input_embeddings()

    # Embed the sentence
    tokenized = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    embedded = word_embeddings(tokenized.input_ids)
    return embedded

def get_verbalized_grads_from_wrapped_model(wrapped_model,
                                            tokenizer,
                                            inputs: str,
                                            loss_fct,
                                            targets,
                                            verbalizer: List[int],
                                            smoothing=0,
                                            query_length=None,
                                            norm=1,
                                            top_k=10,
                                            step_size=1,
                                            gradient_manipulation: str="clipping"
                                            ):
    """
    Calculate cross entropy loss over a subset of the vocabulary.

    Args:
    - logits (torch.Tensor): The predicted logits from the model.
    - targets (torch.Tensor): The target labels.
    - subset_indices (list): List of indices representing the subset of the vocabulary.

    Returns:
    - torch.Tensor: The cross entropy loss.
    """
    def normalize(x):
        t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()

        return x / (t.view(-1, *([1] * x.shape[1:])) + 1e-12)
    def L2_norm(x, keepdim=False):
        z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        if keepdim:
            z = z.view(-1, *[1]*(len(x.shape) - 1))
        return z
    tokenized = tokenizer(inputs, return_tensors="pt", padding=True)
    tokenized["input_ids"] = tokenized["input_ids"].to(wrapped_model.model.device)
    tokenized["attention_mask"] = tokenized["attention_mask"].to(wrapped_model.model.device)
    with torch.enable_grad():
        outputs = wrapped_model(
            # **tokenized,
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            output_hidden_states=True,
        )
        one_hot_dist = torch.zeros(outputs.logits.size(0), outputs.logits.shape[-1])
        one_hot_dist[:, targets[0].cpu().numpy()] = 1
        one_hot_dist = label_smoothing(one_hot_dist, smoothing=smoothing)
        loss = loss_fct(outputs.logits[:, -1, :], one_hot_dist.to(wrapped_model.model.device))
        # yes_token = tokenizer.encode("Yes", add_special_tokens=False)[0]
        # no_token = tokenizer.encode("No", add_special_tokens=False)[0]
        # # one_hot_dist = torch.zeros(outputs.logits.size(0), outputs.logits.shape[-1])
        # # one_hot_dist[:, targets[0].cpu().numpy()] = 1
        # # one_hot_dist = label_smoothing(one_hot_dist, smoothing=smoothing)
        # one_hot_dist = torch.zeros(outputs.logits.size(0), 2)
        # one_hot_dist[:, 1] = 1
        # print(torch.cat([outputs.logits[:, -1, yes_token:yes_token+1], outputs.logits[:, -1, no_token:no_token+1]], dim=-1))
        # loss = loss_fct(torch.cat([outputs.logits[:, -1, yes_token:yes_token+1], outputs.logits[:, -1, no_token:no_token+1]], dim=-1), one_hot_dist.to(wrapped_model.model.device))

        grads = {}
        norms = {}
        hidden_states = outputs.hidden_states[1:] # outputs.hidden_states[0] is the embedding layer
        if gradient_manipulation == "pgd":
            X_pgd = {}
            for i in range(len(hidden_states)):
                X_pgd[i] = hidden_states[i].clone()

        for i in range(len(hidden_states)):
            grads[i] = torch.autograd.grad(loss, hidden_states[i], retain_graph=True, allow_unused=True)[0]
            norms[i] = torch.norm(grads[i], dim=-1, p=2, keepdim=True)
            if gradient_manipulation == "clipping":
                pass
                # norm_mask = norms[i] <= norm
                # temp_norms = norms[i].clone()
                # temp_norms[norm_mask] = 1
                # norms[i][norm_mask] = 1
                # grads[i] = grads[i] / (temp_norms + 1e-12)
            elif gradient_manipulation == "pgd":
                epsilon = 0.2
                eta = step_size * grads[i] / (norms[i] + 1e-12)
                X_pgd[i] = X_pgd[i].data + eta
                X_pgd[i] = torch.clamp(X_pgd[i], hidden_states[i] - epsilon, hidden_states[i] + epsilon)
                grads[i] = X_pgd[i] - hidden_states[i]
            elif gradient_manipulation == "autopgd":
                pass
        norm_tensor = torch.stack([norms[key] for key in norms], dim=0).squeeze(dim=-1)
        save_shape = norm_tensor.shape
        # print(save_shape)
        norm_tensor[:, :, query_length:] = 0
        values, indices = torch.topk(norm_tensor.view(-1), top_k, dim=-1)
        flat_mask = torch.zeros_like(norm_tensor.view(-1))
        flat_mask[indices] = 1
        norm_mask = flat_mask.view(save_shape)
        for i, norm_mask_layer in enumerate(norm_mask):
            norm_mask_layer = norm_mask_layer.unsqueeze(dim=-1)
            # print(grads[i].shape)
            # print(norm_mask_layer.shape)
            grads[i] = grads[i] * norm_mask_layer
            grads[i] = grads[i] / (norms[i] + 1e-12)

        
        # print(grads)
        probs = softmax(outputs.logits[:, -1, verbalizer].detach().cpu().numpy()[0])
        logits = outputs.logits

    return grads, outputs, loss, probs, logits, norms

def get_verbalized_grads(model, tokenizer, inputs: str, loss_fct, targets, verbalizer: List[int]):
    """
    Calculate cross entropy loss over a subset of the vocabulary.

    Args:
    - logits (torch.Tensor): The predicted logits from the model.
    - targets (torch.Tensor): The target labels.
    - subset_indices (list): List of indices representing the subset of the vocabulary.

    Returns:
    - torch.Tensor: The cross entropy loss.
    """
    tokenized = tokenizer(inputs, return_tensors="pt", padding=True)
    tokenized["input_ids"] = tokenized["input_ids"].to(model.model.device)
    tokenized["attention_mask"] = tokenized["attention_mask"].to(model.model.device)
    outputs = model(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        output_hidden_states=True,
    )
    loss = loss_fct(outputs.logits[:, -1, :], targets)
    # print(f"Loss: {loss}")

    grads = {}
    norms = {}
    hidden_states = outputs.hidden_states[1:] # outputs.hidden_states[0] is the embedding layer
    for i in range(len(hidden_states)):
        grads[i] = torch.autograd.grad(loss, hidden_states[i], retain_graph=True, allow_unused=True)[0]
        norms[i] = torch.norm(grads[i], dim=-1, keepdim=True)
        norm_mask = norms[i] <= 1
        norms[i][norm_mask] = 1
        grads[i] = grads[i] / norms[i]

    probs = softmax(outputs.logits[:, -1, verbalizer].detach().cpu().numpy()[0])
    logits = outputs.logits

    return grads, hidden_states, loss, probs, logits, norms

def control_on_layers(layer_ids, wrapped_model, grads, query_length, token_pos="start"):
    """
    Control the activations of the model on the specified layers.
    """
    wrapped_model.unwrap()

    block_name="decoder_block"

    wrapped_model.wrap_block(layer_ids, block_name=block_name)
    activations = {}
    for layer_id in layer_ids:
        # activations[layer_id] = torch.tensor(coeff * grads[layer_id]).to(model.device).half()
        if isinstance(token_pos, str):
            if token_pos == "start":
                activations[layer_id] = grads[layer_id][:, :query_length, :]
            elif token_pos == "full":
                activations[layer_id] = grads[layer_id][:, :, :]
                token_pos = "start"
            elif token_pos == "end":
                activations[layer_id] = grads[layer_id][:, -query_length:, :]
        elif isinstance(token_pos, int):
            activations[layer_id] = grads[layer_id][:, token_pos, :].unsqueeze(dim=1)
        elif isinstance(token_pos, list):
            print("using list")
            activations[layer_id] = grads[layer_id][:, :, :]

        wrapped_model.set_controller(layer_id, activations[layer_id], token_pos=token_pos, masks=1)

    return wrapped_model

def get_dual_grads(model, tokenizer, input_list: List[str], loss_fct, verbalizer: List[int]):
    """
    1. Mix the distribution of a pair of contrastive prompts using max-pooling (union)
    2. Compute the cross entropy loss between the new distribution and the normalized target distribution, i.e. normalized probs for each target token (verbalizer)

    I'm currently doing two forward passes to avoid using padding tokens in the loss computation.
    """
    def get_sentence_embedding(model, tokenizer, sentence):
        sentence = sentence.strip().replace('"', "")
        word_embeddings = model.get_input_embeddings()

        # Embed the sentence
        tokenized = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
        embedded = word_embeddings(tokenized.input_ids)
        return embedded
    pos_ground_truth_embeds = get_sentence_embedding(
        model, tokenizer, input_list[0]
    )
    neg_ground_truth_embeds = get_sentence_embedding(
        model, tokenizer, input_list[1]
    )
    pos_outputs = model(
        inputs_embeds=pos_ground_truth_embeds,
        # input_ids=input_list[0]["input_ids"],
        # attention_mask=input_list[0]["attention_mask"],
        output_hidden_states=True,
    )
    neg_outputs = model(
        inputs_embeds=neg_ground_truth_embeds,
        # input_ids=input_list[1]["input_ids"],
        # attention_mask=input_list[1]["attention_mask"],
        output_hidden_states=True,
    )

    logits = torch.cat([pos_outputs.logits[:, -1, verbalizer[0]], neg_outputs.logits[:, -1, verbalizer[1]]], dim=0)
    # loss = loss_fct(outputs.logits[:, -1, :], targets)
    mixed_logits = logits
    target_dist = torch.ones_like(mixed_logits)
    # target_dist[:, verbalizer] = 1
    target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)
    # loss = loss_fct(mixed_logits, target_dist)
    pos_loss = loss_fct(pos_outputs.logits[:, -1, :], verbalizer[0] * torch.ones(1).long().to(model.device))
    neg_loss = loss_fct(neg_outputs.logits[:, -1, :], verbalizer[1] * torch.ones(1).long().to(model.device))

    pos_prob = probs = softmax(pos_outputs.logits[:, -1, verbalizer].detach().cpu().numpy()[0])[0]
    neg_prob = probs = softmax(neg_outputs.logits[:, -1, verbalizer].detach().cpu().numpy()[0])[1]
    probs = [pos_prob, neg_prob]
    # print(f"Loss: {loss}")

    pos_grads = {}
    pos_norms = {}
    neg_grads = {}
    neg_norms = {}
    for i in range(len(pos_outputs.hidden_states)):
        pos_grads[i] = torch.autograd.grad(pos_loss, pos_outputs.hidden_states[i], retain_graph=True)[0]
        pos_norms[i] = torch.norm(pos_grads[i], dim=-1, keepdim=True)
        norm_mask = pos_norms[i] <= 1
        pos_norms[i][norm_mask] = 1
        pos_grads[i] = pos_grads[i] / pos_norms[i]
        # pos_grads[i] = pos_grads[i] / pos_norms[i]

    for i in range(len(neg_outputs.hidden_states)):
        neg_grads[i] = torch.autograd.grad(neg_loss, neg_outputs.hidden_states[i], retain_graph=True)[0]
        neg_norms[i] = torch.norm(neg_grads[i], dim=-1, keepdim=True)
        norm_mask = neg_norms[i] <= 1
        neg_norms[i][norm_mask] = 1
        neg_grads[i] = neg_grads[i] / neg_norms[i]
        # neg_grads[i] = neg_grads[i] / neg_norms[i]

    return pos_loss+neg_loss, probs, pos_grads, neg_grads


def vanilla_control(model: AutoModelForCausalLM,
                    tokenizer: AutoTokenizer,
                    wrapped_model,
                    inputs: str,
                    target: torch.Tensor,
                    query_length: int,
                    verbalizer: List[int],
                    acc_grads: Dict,
                    loss_fct: torch.nn.CrossEntropyLoss,
                    coeff: float=0.05,
                    **kwargs):
    """
    Control with a single suffix

    Args:
        - model: model to be controlled
        - wrapped_model: wrapped model to be controlled
        - inputs: input string with suffix
        - target: target token
        - query_length: length of the input query
        - verbalizer: list of target tokens ["Yes", "No"] by default
        - acc_grads: accumulated gradients
        - coeff: coefficient for control
    """
    wrapped_model.reset()
    # print(inputs)
    grads, outputs, loss, probs, logits, norms = get_verbalized_grads(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        loss_fct=loss_fct,
        targets=target,
        verbalizer=verbalizer
    )
    for i in grads:
        if i in acc_grads:
            min_len = min(acc_grads[i].size(1), grads[i].size(1))
            acc_grads[i] = acc_grads[i][:, :min_len] + coeff * grads[i][:, :min_len]
        else:
            acc_grads[i] = coeff * grads[i]

    token_pos = "start"     # control on input tokens by default
    layer_ids = list(range(0, 32, 1))   # control on all layers by default
    wrapped_model = control_on_layers(
        layer_ids=layer_ids,
        wrapped_model=wrapped_model,
        grads=acc_grads,
        query_length=query_length,
        token_pos=token_pos,
    )
    return wrapped_model, acc_grads, loss, probs

def search_step_size(orig_input: str,
                                suffix: str,
                                wrapped_model,
                                acc_grads: Dict={},
                                random_seed=0,
                                layer_ids: List[int]=list(range(0, 32, 1)),
                                smoothing=0,
                                initial_step_size: float=0.1,
                                loss_threshold: float=1e-5,
                                max_iterations: int=3,
                                scale_factor: float=0.5,
                                initial_grads_loss: Dict=None,
                                do_sample=False,
                                verbose=False,
                                gradient_manipulation="clipping",
                                **control_args
                                ) -> float:
    """
    A search algorithm to find an optimal step-size that minimizes the loss function.
    
    Params:
        orig_input: The original input sentence
        suffix: The suffix to be added to the input sentence
        initial_step_size: The starting step-size for the search.
        loss_threshold: The loss value to achieve before stopping the search.
        max_iterations: The maximum number of iterations to run the search.
        scale_factor: The factor by which to scale the step-size on each iteration.
        attack_config: The configuration for the generation exploitation attack.
        
    Return:
        The best step size
    """
    query_length = control_args.pop("query_length")
    tokenizer = control_args.pop("tokenizer")
    loss_fct = control_args.pop("loss_fct")
    target = control_args.pop("target")
    verbalizer = control_args.pop("verbalizer")

    # Initialize variables
    best_loss = float('inf')
    best_step_size = initial_step_size
    current_step_size = initial_step_size
    
    # grads, outputs, loss, probs, logits, norms = get_verbalized_grads_from_wrapped_model(
    #     inputs=input_with_suffix,
    #     wrapped_model=model,
    #     tokenizer=tokenizer,
    #     loss_fct=loss_fct,
    #     targets=target,
    #     verbalizer=verbalizer
    # )

    # del outputs, probs, logits, norms
    input_with_suffix = initial_grads_loss["controlled_output"]
    loss = initial_grads_loss["loss"]
    grads = initial_grads_loss["grads"]

    if verbose:
        print(f"Input w/ suffix: {input_with_suffix}")
        print(f"Initial Loss: {loss}")

    for i in range(max_iterations):
        test_step_size = current_step_size

        temp_grads = {}
        for i in grads:
            if i in acc_grads:
                temp_grads[i] = acc_grads[i][:, :query_length] + test_step_size * grads[i][:, :query_length]    # should always use the initial grads
            else:
                temp_grads[i] = test_step_size * grads[i][:, :query_length]
        
        token_pos = "start"     # control on input tokens by default
        wrapped_model = control_on_layers(
            layer_ids=layer_ids,
            wrapped_model=wrapped_model,
            grads=temp_grads,
            query_length=query_length,
            token_pos=token_pos,
        )
        input_with_suffix = [input + suffix for input in wrapped_model.generate(**orig_input, do_sample=do_sample, **control_args)]
        wrapped_model.reset()
        _, outputs, loss, probs, logits, norms = get_verbalized_grads_from_wrapped_model(
            inputs=input_with_suffix,
            wrapped_model=wrapped_model,
            tokenizer=tokenizer,
            loss_fct=loss_fct,
            targets=target,
            verbalizer=verbalizer,
            smoothing=smoothing,
            gradient_manipulation=gradient_manipulation,
        )
        if verbose:
            print(f"Input w/ suffix: {input_with_suffix}")
            print(f"Loss: {loss}")

        del outputs, probs, logits, norms
        
        # Check if the loss is better than what we have seen so far
        if loss < best_loss:
            best_loss = loss
            best_step_size = test_step_size
            
            # Check if the loss is below the threshold
            if loss <= loss_threshold:
                # print(f"Better Step-size found: {best_step_size}, Loss: {loss}")
                return best_step_size
        else:
            # print(f"Step-size found: {test_step_size}, Loss: {loss}")
            pass
    
        # If not, scale down the absolute value of the step-size and continue
        current_step_size *= scale_factor
    if verbose:
        print(f"Best step-size found: {best_step_size}, Loss: {best_loss}")
    return best_step_size, best_loss


def KL_divergence(p, q, epsilon=1e-9):
    """Compuates KL divergence between two probability distributions

    Args:
        p (torch.tensor): probability distribution
        q (torch.tensor): probability distribution

    Returns:
        float: KL divergence
    """
    return torch.sum(p * torch.log((p + epsilon) / (q + epsilon)))


def label_smoothing(one_hot_labels, smoothing=0.5):
    """
    Applies label smoothing to one-hot labels.

    Args:
        one_hot_labels (np.ndarray): One-hot encoded labels with shape (batch_size, num_classes).
        smoothing (float): Smoothing factor between 0 and 1.

    Returns:
        np.ndarray: Smoothed labels.
    """
    num_classes = one_hot_labels.shape[1]
    smooth_labels = (1.0 - smoothing) * one_hot_labels + (smoothing / num_classes) * np.ones_like(one_hot_labels)
    return smooth_labels