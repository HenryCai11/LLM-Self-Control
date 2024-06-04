import torch
from torch.autograd import Variable
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Union
from copy import deepcopy
from scipy.special import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import kl_div
import numpy as np
from self_control.utils.suffix_manager import SuffixItem

from torch import nn
from torch.nn import functional as F
from torch.func import functional_call, vmap
from transformers import LlamaForCausalLM, MistralForCausalLM
import plotly.graph_objects as go
import random

def get_suffix_grads_from_wrapped_model(wrapped_model,
                                            tokenizer,
                                            inputs: str,
                                            targets,
                                            contrastive_paris: List[int],
                                            loss_fct=nn.CrossEntropyLoss(),
                                            smoothing=0,
                                            query_length=None,
                                            norm=1,
                                            top_k=20,
                                            step_size=1,
                                            gradient_manipulation: str="clipping",
                                            binary=False,
                                            temperature=10,
                                            ):
    """
    Calculate cross entropy loss over a subset of the vocabulary.

    Args:
        - tokenizer
        - inputs
        - targets
        - contrastive_paris


    Returns:
    - torch.Tensor: The cross entropy loss.
    """
    tokenized = tokenizer(inputs, return_tensors="pt", padding=True)
    tokenized["input_ids"] = tokenized["input_ids"].to(wrapped_model.model.device)
    tokenized["attention_mask"] = tokenized["attention_mask"].to(wrapped_model.model.device)
    pos_token = tokenizer.encode(contrastive_paris[0], add_special_tokens=False)[0]
    neg_token = tokenizer.encode(contrastive_paris[1], add_special_tokens=False)[0]
    assert targets[0] in [pos_token, neg_token]
    with torch.enable_grad():
        outputs = wrapped_model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            output_hidden_states=True,
        )
        if not binary:
            one_hot_dist = torch.zeros(outputs.logits.size(0), outputs.logits.shape[-1])
            one_hot_dist[:, targets[0].cpu().numpy()] = 1
            one_hot_dist = label_smoothing(one_hot_dist, smoothing=smoothing)
            loss = loss_fct(outputs.logits[:, -1, :], one_hot_dist.to(wrapped_model.model.device))
        elif binary:
            # only consider that all the targets are the same for now
            # TODO: think about if this is the best to feed contrastive pairs
            if targets[0] == pos_token:
                loss = torch.sum(-1 / (1 + torch.exp(-(outputs.logits[:, -1, pos_token] / temperature - outputs.logits[:, -1, neg_token] / temperature))))
            elif targets[0] == neg_token:
                loss = torch.sum(-1 / (1 + torch.exp(-(outputs.logits[:, -1, neg_token] / temperature - outputs.logits[:, -1, pos_token] / temperature))))
            else:
                raise ValueError(f"Unknown {targets[0]}")


        grads = {}
        norms = {}
        hidden_states = outputs.hidden_states[1:] # outputs.hidden_states[0] is the embedding layer
        orig_norm = torch.norm(torch.stack([hidden_state for hidden_state in hidden_states]), p=2, dim=-1)
        if gradient_manipulation == "pgd":
            X_pgd = {}
            for i in range(len(hidden_states)):
                X_pgd[i] = hidden_states[i].clone()

        for i in range(len(hidden_states)):
            grads[i] = torch.autograd.grad(loss, hidden_states[i], retain_graph=True, allow_unused=True)[0]
            norms[i] = torch.norm(grads[i], dim=-1, p=2, keepdim=True)
            if gradient_manipulation == "clipping":
                pass
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
        norm_tensor[:, :, query_length:] = 0
        # TODO: pgd
        if top_k > 0:
            values, indices = torch.topk(norm_tensor.view(-1), top_k, dim=-1)
            flat_mask = torch.zeros_like(norm_tensor.view(-1))
            flat_mask[indices] = 1
            norm_mask = flat_mask.view(save_shape)
            for i, norm_mask_layer in enumerate(norm_mask):
                norm_mask_layer = norm_mask_layer.unsqueeze(dim=-1)
                normalize_mask = norms[i] <= norm
                temp_norms = norms[i].clone()
                temp_norms[normalize_mask] = 1

                grads[i] = grads[i] * norm_mask_layer
                grads[i] = grads[i] / (temp_norms + 1e-12)
        else:
            for i in grads:
                normalize_mask = norms[i] <= norm
                temp_norms = norms[i].clone()
                temp_norms[normalize_mask] = 1

                grads[i] = grads[i] / (temp_norms + 1e-12)


        wrapped_model.reset()
        cleaned_outputs = wrapped_model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            output_hidden_states=True,
        )
        ret_prob_list = []
        if targets[0] == pos_token:
            ret_probs = 1 / (1 + torch.exp(-(cleaned_outputs.logits[:, -1, pos_token] / temperature - cleaned_outputs.logits[:, -1, neg_token] / temperature))).detach().cpu().numpy()
        elif targets[0] == neg_token:
            ret_probs = 1 / (1 + torch.exp(-(cleaned_outputs.logits[:, -1, neg_token] / temperature - cleaned_outputs.logits[:, -1, pos_token] / temperature))).detach().cpu().numpy()
        ret_prob_list.append(float(ret_probs))
        logits = outputs.logits

    return grads, outputs, loss, ret_prob_list, logits, norms, orig_norm


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


def search_step_size(orig_input: Dict,
                                wrapped_model,
                                suffix: Union[SuffixItem, List[SuffixItem]],
                                acc_grads: Dict={},
                                layer_ids: List[int]=list(range(0, 32, 1)),
                                smoothing=0,
                                top_k=10,
                                random_seed=0,
                                n_branches=6,
                                initial_step_size: float=0.1,
                                max_iterations: int=3,
                                scale_factor: float=2,
                                initial_grads_loss: Dict=None,
                                do_sample=False,
                                verbose=False,
                                search_threshold: float=0,
                                gradient_manipulation="clipping",
                                **control_args
                                ) -> float:
    """
    A search algorithm to find an optimal step-size that minimizes the loss function.
    
    Params:
        - orig_input: The original input sentence
        - suffix: The suffix to be added to the input sentence
        - initial_step_size: The starting step-size for the search.
        - max_iterations: The maximum number of iterations to run the search.
        - scale_factor: The factor by which to scale the step-size on each iteration.
        
    Return:
        The best step size
    """
    wrapped_model.reset()
    query_length = control_args.pop("query_length")
    tokenizer = control_args.pop("tokenizer")
    loss_fct = control_args.pop("loss_fct")
    target = control_args.pop("target")
    contrastive_paris = control_args.pop("contrastive_paris")

    max_new_tokens = control_args.get("max_new_tokens", 50)

    input_with_suffix = initial_grads_loss["controlled_output"]
    score = initial_grads_loss["score"]
    grads = initial_grads_loss["grads"]

    # Initialize variables
    best_score = score
    best_verbose_scores = []
    print(f"Initial Score {best_score}")
    best_step_size = initial_step_size
    current_step_size = initial_step_size

    verbose_output = []

    if verbose:
        print(f"Input w/ suffix: {input_with_suffix}")

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
        verbose_scores = []
        if isinstance(suffix, list):
            composed_grads = {}
            multi_score = 0
            for suffix_item in suffix:
                suffix_string = suffix_item.suffix
                target = suffix_item.target
                target_token = tokenizer.encode(target, add_special_tokens=False, return_tensors='pt').squeeze(0)
                assert target_token.shape[-1] == 1, "Target should be a single token for now."
                target_token = (target_token * torch.ones(1).long()).to(wrapped_model.model.device)
                contrastive_paris = [target_token[0]]
                responses = wrapped_model.generate(**orig_input, do_sample=do_sample, **control_args)
                input_list = [input + suffix_string for input in responses]
                wrapped_model.reset()
                grads, outputs, loss, probs, logits, norms, orig_norm = get_suffix_grads_from_wrapped_model(
                    wrapped_model=wrapped_model,
                    tokenizer=tokenizer,
                    inputs=input_list,
                    loss_fct=loss_fct,
                    targets=target_token,
                    contrastive_paris=contrastive_paris,
                    smoothing=smoothing,
                    top_k=top_k,
                    query_length=query_length,
                    norm=1,
                    gradient_manipulation=gradient_manipulation,
                )
                multi_score += sum(probs)
                verbose_scores.append(sum(probs))
                # FIXME: fix the hard-coded normalization
                composed_grads = {k: (composed_grads[k][:, :query_length] + grads[k][:, :query_length] * suffix_item.direction * 0.5) if k in composed_grads else grads[k][:, :query_length] * 0.5\
                                    for k in set(grads)}
            grads = composed_grads
            del composed_grads
            score = multi_score / len(suffix)
        else:
            # responses = wrapped_model.generate(**orig_input, do_sample=do_sample, **control_args)
            # input_with_suffix = [input + suffix.suffix for input in responses]
            # wrapped_model.reset()
            responses, score_list = wrapped_model.suffix_decoding(input_ids=orig_input["input_ids"], 
                                                        attention_mask=orig_input["attention_mask"],
                                                        n_branches=n_branches,
                                                        suffix=suffix, 
                                                        max_new_tokens=max_new_tokens,
                                                        verbose=verbose,
                                                        record=False,
                                                        )

            score = sum(score_list) / len(score_list)
            verbose_scores.append(score)
        if verbose:
            print(f"Input w/ suffix: {input_with_suffix}")
            print(f"Score: {score}")
        verbose_output.append({
            "phase": "M-step",
            "response": responses[0],
            "score": score,
            "step_size": test_step_size,
        })
        del responses
        
        # adjust this threshold if needed
        if score - best_score > search_threshold:
            best_score = score
            best_verbose_scores = verbose_scores
            best_step_size = test_step_size
            return best_step_size, best_score, best_verbose_scores, verbose_output, temp_grads
        else:
            pass
    
        # If not, scale down the absolute value of the step-size and continue
        current_step_size *= scale_factor
    if verbose:
        print(f"No best step-size found")
    # return best_step_size, best_loss
    return 0, best_score, best_verbose_scores, verbose_output, {}


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


def greedy_decode(model, tokenizer, input_ids, max_length=50):
    """
    The generation utility for Prefix Controller
    """
    def token_id_to_embedding(token_id):
        return model.base_model.model.model.embed_tokens(token_id.to(model.device))
    dot_token_ids = [tokenizer.convert_tokens_to_ids(".")]
    prefix_token_ids = tokenizer.encode("<<SYS>> You are an assistant <</SYS>>", add_special_tokens=False)
    # prefix_input_ids = torch.tensor(prefix_token_ids + dot_token_ids * 5).unsqueeze(dim=0)
    prefix_input_ids = torch.arange(len(prefix_token_ids + dot_token_ids * 5)).unsqueeze(dim=0)
    bos_token = torch.tensor([tokenizer.bos_token_id]).unsqueeze(dim=0)
    prefix_input_ids = torch.cat([bos_token, prefix_input_ids], dim=-1)
    prefix_mask = torch.ones_like(prefix_input_ids)
    attention_mask = torch.ones_like(input_ids)

    if isinstance(model.base_model.model, LlamaForCausalLM):
        input_embeds = torch.cat([model.prefix_embedder(prefix_input_ids).to(model.device), model.base_model.model.model.embed_tokens(input_ids.to(model.device))], dim=1)
        attention_mask = torch.cat([prefix_mask.to(model.device), attention_mask.to(model.device)], dim=-1)
    elif isinstance(model.base_model.model, MistralForCausalLM):
        input_embeds = torch.cat([model.prefix_embedder(prefix_input_ids).to(model.device), model.base_model.model.model.embed_tokens(input_ids.to(model.device))], dim=1)
        attention_mask = torch.cat([prefix_mask.to(model.device), attention_mask.to(model.device)], dim=-1)
    gen_ids = []
    EOS_TOKEN_ID = tokenizer.eos_token_id
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(
                inputs_embeds=input_embeds.to(torch.bfloat16),
                attention_mask=attention_mask
            )
            # Assume outputs are logits from the final layer
            predictions = outputs.logits[:, -1, :]  # Get the logits for the last token output
            predicted_token_id = torch.argmax(predictions, dim=-1).unsqueeze(-1)  # Most likely next token

            gen_ids.append(int(predicted_token_id[0][0].cpu()))
            # Assuming you have a method to convert token_ids to embeddings
            next_token_embeds = token_id_to_embedding(predicted_token_id)
            
            # Append the predicted token embeddings for the next round of inputs
            input_embeds = torch.cat((input_embeds, next_token_embeds), dim=1)

            # Check for stopping criteria here, e.g., if predicted_token_id is an EOS token
            if predicted_token_id == EOS_TOKEN_ID:
                gen_ids = gen_ids[:-1]
                break
    return tokenizer.decode(gen_ids)


def get_prefix_input_ids(tokenizer, prompt_type="default") -> torch.Tensor:
    """
    Customize your prompt for the Prefix Controller here
    """
    if prompt_type == "default":
        # We concat the prefix and the input in the collate_fn
        dot_token_ids = [tokenizer.convert_tokens_to_ids(".")]
        prefix_token_ids = tokenizer.encode("<<SYS>> You are an assistant <</SYS>>", add_special_tokens=False)
        prefix_input_ids = torch.arange(len(prefix_token_ids + dot_token_ids * 5)).unsqueeze(dim=0)
        bos_token = torch.tensor([tokenizer.bos_token_id]).unsqueeze(dim=0)
        prefix_input_ids = torch.cat([bos_token, prefix_input_ids], dim=-1)
    else:
        raise ValueError(f"Prompt type {prompt_type} not defined")
    
    return prefix_input_ids
    

def display_responses(responses, scores):
    """
    Displays a formatted list of responses with their scores when verbose is True.

    Args:
        - responses (List[str]): A list of responses
        - scores (List[float]): A list of scores for the corresponding responses
    """
    print("Testing")
    # Display header
    print(f"{'Response':<20} | {'Score':>5}")
    print("-" * 28)  # Header separator
    
    # Display each response and its score
    for (response, score) in zip(responses, scores):
        print(f"{response:<20} | {score:5}")


import textwrap

def draw_gen_trajectory(iteration_data, suffix, output_name="output"):
    iterations_data = iteration_data
    suffix_used = suffix.suffix

    # Create a figure
    fig = go.Figure()

    # Horizontal offset for E-step and M-step
    offset = 0.1

    # Text wrapping function
    def wrap_text(text, width=50):
        return '<br>'.join(textwrap.wrap(text, width))

    # Adding traces for each branch
    for data in iterations_data:
        iteration = data['iteration']
        for branch in data['branches']:
            position = iteration - offset if branch['phase'] == 'M-step' else iteration + offset
            response_text = wrap_text(branch['response'])
            hovertext = f"{response_text}<br><b>Score:</b> {branch['score']}" + \
                        (f"<br><b>Step-size:</b> {branch['step_size']}" if 'step_size' in branch else "")
            fig.add_trace(go.Scatter(
                x=[position], y=[branch['score']],
                mode='markers+text',
                text=[' '.join(branch['response'].split()[10:13]) + '...'],
                hovertext=hovertext,
                hoverinfo='text',
                name=f"{branch['phase']} {iteration}",
                marker=dict(symbol='circle' if branch['phase'] == 'E-step' else 'x', size=12)
            ))

    # Update layout with a more descriptive title and set figure dimensions
    fig.update_layout(
        title=f'Trajectory of Responses and Suffix Scores Over Iterations<br><sup>Suffix: {suffix_used}</sup>',
        xaxis_title='Iteration',
        yaxis_title='Suffix Score',
        hovermode='closest',
        legend_title="Phase/Iteration",
        width=1400,  # Adjusted width to fit more content
        height=600,
        margin=dict(l=50, r=50, t=100, b=50)
    )

    # Save the figure as an interactive HTML file
    fig.write_html(f"{output_name}.html")

    # Show the plot
    fig.show()