# Original code from Representation Engineering
# wrapping classes
import torch
import warnings
import numpy as np
from self_control.utils import get_sentence_embedding, get_verbalized_grads_from_wrapped_model, control_on_layers, label_smoothing, search_step_size
from self_control.utils.suffix_manager import SuffixItem
from scipy.special import softmax
from typing import Union, List
from peft import PeftModel

import transformers
import random
from copy import deepcopy

class WrappedBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.output = None
        self.controller = None
        self.mask = None
        self.token_pos = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        if isinstance(output, tuple):
            self.output = output[0]
            modified = output[0]
        else:
            self.output = output
            modified = output
            
        if self.controller is not None:
            if self.mask is not None:
                mask = self.mask

            # we should ignore the padding tokens when doing the activation addition
            # mask has ones for non padding tokens and zeros at padding tokens.
            # only tested this on left padding
            elif "position_ids" in kwargs:
                pos = kwargs["position_ids"]
                zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
                col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
                target_shape = modified.shape
                mask = (col_indices >= zero_indices).float().reshape(target_shape[0], target_shape[1], 1)
                mask = mask.to(modified.dtype)
            else:
                # print(f"Warning: block {self.block_name} does not contain information 'position_ids' about token types. When using batches this can lead to unexpected results.")
                mask = 1.0

            if len(self.controller.shape) == 1:
                self.controller = self.controller.reshape(1, 1, -1)
            assert len(self.controller.shape) == len(modified.shape), f"Shape of controller {self.controller.shape} does not match shape of modified {modified.shape}."

            self.controller = self.controller.to(modified.device)
            if type(mask) == torch.Tensor:
                mask = mask.to(modified.device)
            if isinstance(self.token_pos, int):
                try:
                    modified[:, self.token_pos] = modified[:, self.token_pos] + self.controller * mask
                except:
                    warnings.warn(f"Control Pos {self.token_pos} is out of the bound, make sure you are aware of this behavior")
            elif isinstance(self.token_pos, list) or isinstance(self.token_pos, tuple) or isinstance(self.token_pos, np.ndarray):
                # print(modified.shape)
                # print(self.controller.shape)
                modified[:, self.token_pos] = modified[:, self.token_pos] + self.controller[:, self.token_pos] * mask
            elif isinstance(self.token_pos, str):
                if self.token_pos == "end":
                    len_token = self.controller.shape[1]
                    modified[:, -len_token:] = modified[:, -len_token:] + self.controller * mask
                elif self.token_pos == "start":
                    # print(modified.shape)
                    # print(self.controller.shape)
                    len_token = min(self.controller.shape[1], modified.shape[1])
                    if len_token != 1: # In this way can we use use_cache TODO: make this more elegant
                        modified[:, :len_token] = modified[:, :len_token] + self.controller[:, :len_token] * mask
                else:
                    assert False, f"Unknown token position {self.token_pos}."
            else:
                modified = modified + self.controller * mask
        
        # modified = torch.clamp(modified, output[0]-0.1, output[0]+0.1)
        if isinstance(output, tuple):
            output = (modified,) + output[1:] 
        else:
            output = modified
        
        return output

    def set_controller(self, activations, token_pos=None, masks=None):
        self.controller = activations
        self.mask = masks
        self.token_pos = token_pos
        
    def reset(self):
        self.output = None
        self.controller = None
        self.mask = None

    def set_masks(self, masks):
        self.mask = masks

    
class WrappedReadingVecModel(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        if isinstance(model, PeftModel):
            self.model = model.base_model.model
        else:
            self.model = model
        self.tokenizer = tokenizer
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def control_on_layers(self,
                        layer_ids,
                        grads,
                        query_length,
                        token_pos="start",
                        block_name="decoder_block",
                        gradient_manipulation: str="clipping",
                        epsilon: float=0.3
                        ) -> None:
        """
        Control the activations of the model on the specified layers.
        """
        self.unwrap()

        self.wrap_block(layer_ids, block_name=block_name)
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

            self.set_controller(layer_id, activations[layer_id], token_pos=token_pos, masks=1)


    def controlled_generate(self,
                            prompt: List[str]=None,
                            input_ids=None,
                            attention_mask=None,
                            suffix: Union[SuffixItem, List[SuffixItem]]=None,
                            loss_fct=None,
                            verbalizer=None,
                            coeff=-0.1,
                            iterations=5,
                            top_k=20,
                            max_search_steps=3,
                            token_pos="start",
                            layer_ids=list(range(0, 32, 1)),
                            random_seed=0,
                            consistent=True,
                            use_last=False,
                            use_cache=False,
                            smoothing: float = 0,
                            search=False,
                            verbose=False,
                            gradient_manipulation="clipping",
                            return_intermediate=False,
                            return_all_grads=False,
                            return_hiddens=False,
                            return_grads=False,
                            return_logits=False,
                            return_ids=False,
                            remain_control=False,
                            load_best_last=False,
                            last_max_new_tokens=None,
                            do_sample=False,
                            norm=1,
                            epsilon=0.3,
                            annealing: float=1,
                            **kwargs    # for the generate function
                            ):
        """
        
        """
        transformers.set_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        self.model.eval()
        self.reset()
        best_loss = float('inf')
        final_grads = {}
        grad_list = []
        temp_grads = {}
        acc_grads = {}  # accumulated gradients
        best_grads = {}
        gradient_bs = 1 # TODO: default to 1
        orig_coeff = coeff
        final_output_dict = {}

        # Prepare inputs
        inputs = {}
        if prompt is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
            inputs["attention_mask"] = inputs["attention_mask"].to(self.model.device)
            query_length = inputs["input_ids"].size(1)
        else:
            inputs["input_ids"] = input_ids
            inputs["attention_mask"] = attention_mask
            query_length = input_ids.size(1) if len(input_ids.shape) == 2 else input_ids.size(0) # size(0) might be the batch size
        if return_intermediate:
            iterative_outputs = []
        controlled_output = self.generate(**inputs, use_cache=use_cache, do_sample=do_sample, **kwargs) # the original output
        original_output = deepcopy(controlled_output)
        if verbose:
            print("Coeff: ", orig_coeff)
        #     print("Original Output:\n", controlled_output)
        #     rationale = self.generate([output+suffix.suffix for output in controlled_output], use_cache=use_cache, do_sample=do_sample, **kwargs)
        #     print("Rationale:\n", rationale)
        #     print("="*50)

        for iter in range(iterations):
            # print(controlled_output)
            for i in range(len(controlled_output)):
                controlled_output[i] = controlled_output[i] + suffix.suffix

            if isinstance(suffix, list):
                warnings.warn(f"Accepting a list of suffixes has not been tested right now")
                composed_grads = {}
                for suffix_item in suffix:
                    target = suffix_item.target
                    target_token = self.tokenizer.encode(target, add_special_tokens=False, return_tensors='pt').squeeze(0)
                    assert target_token.shape[-1] == 1, "Target should be a single token for now."
                    target_token = (target_token * torch.ones(gradient_bs).long()).to(self.model.device)
                    verbalizer = [target_token[0]]
                    grads, outputs, loss, probs, logits, norms = get_verbalized_grads_from_wrapped_model(
                        wrapped_model=self,
                        tokenizer=self.tokenizer,
                        inputs=controlled_output,
                        loss_fct=loss_fct,
                        targets=target_token,
                        verbalizer=verbalizer,
                        smoothing=smoothing,
                        top_k=top_k,
                        norm=norm,
                        step_size=orig_coeff,
                        gradient_manipulation=gradient_manipulation,
                    )
                    composed_grads = {k: (composed_grads[k] + grads[k] * suffix_item.direction) if k in composed_grads else grads[k]\
                                       for k in set(grads)}
                grads = composed_grads
                del composed_grads
            else:
                target = suffix.target
                target_token = self.tokenizer.encode(target, add_special_tokens=False, return_tensors='pt').squeeze(0)
                if target_token.shape[-1] != 1:
                    warnings.warn(f"Target should be single token for now. Using the first token of suffix {suffix.target} as target")
                target_token = (target_token * torch.ones(gradient_bs).long()).to(self.model.device)
                verbalizer = [target_token[0]]
                grads, outputs, loss, probs, logits, norms = get_verbalized_grads_from_wrapped_model(
                    wrapped_model=self,
                    tokenizer=self.tokenizer,
                    inputs=controlled_output,
                    loss_fct=loss_fct,
                    targets=target_token,
                    verbalizer=verbalizer,
                    smoothing=smoothing,
                    top_k=top_k,
                    norm=norm,
                    step_size=orig_coeff,
                    gradient_manipulation=gradient_manipulation,
                )

            if loss < best_loss:
                best_loss = loss
                best_grads = acc_grads
            if search:
                step_size = search_step_size(
                    orig_input              =   prompt,
                    suffix                  =   suffix.suffix,
                    wrapped_model           =   self,
                    acc_grads               =   acc_grads,
                    initial_step_size       =   orig_coeff,
                    verbose                 =   verbose,
                    max_iterations          =   max_search_steps,
                    smoothing               =   smoothing,
                    top_k                   =   top_k,
                    initial_grads_loss      =   {
                        "grads": grads,
                        "loss": loss,
                        "controlled_output":    controlled_output
                    },
                    gradient_manipulation   =   gradient_manipulation,
                    # control args
                    tokenizer               =   self.tokenizer,
                    target                  =   target_token,
                    query_length            =   query_length,
                    verbalizer              =   verbalizer,
                    loss_fct                =   loss_fct,
                    **kwargs
                )
                coeff = step_size
                if gradient_manipulation == "pgd":  # TODO: optimize this
                    grads, outputs, loss, probs, logits, norms = get_verbalized_grads_from_wrapped_model(
                        wrapped_model=self,
                        tokenizer=self.tokenizer,
                        inputs=controlled_output,
                        loss_fct=loss_fct,
                        targets=target_token,
                        verbalizer=verbalizer,
                        smoothing=smoothing,
                        top_k=top_k,
                        norm=norm,
                        step_size=coeff,
                        gradient_manipulation=gradient_manipulation,
                    )
            if gradient_manipulation == "pgd":  # If pgd then it's already controlled with the step size
                coeff = 1
            for i in grads:
                if i in acc_grads:
                    acc_grads[i] = acc_grads[i][:, :query_length] + coeff * grads[i][:, :query_length]
                else:
                    acc_grads[i] = coeff * grads[i][:, :query_length]
            coeff *= annealing
            if return_all_grads:
                temp_grads = {}
                for i in acc_grads:
                    temp_grads[i] = acc_grads[i].detach().cpu().clone()
                grad_list.append(temp_grads)
                del temp_grads
            self.control_on_layers(
                layer_ids=layer_ids,
                grads=acc_grads,
                query_length=query_length,
                token_pos=token_pos,
                gradient_manipulation=gradient_manipulation,
                epsilon=epsilon
            )

            controlled_output = self.generate(**inputs, use_cache=use_cache, do_sample=do_sample, **kwargs)
            if not consistent:
                self.reset()
            # if verbose:
            #     print(f"Loss from the iteration {iter}: {loss.item()}")
            #     print(f"Output form the iteration {iter}:\n", controlled_output)
            #     rationale = self.generate([output + suffix.suffix for output in controlled_output], keep_input=False, use_cache=use_cache, do_sample=do_sample, **kwargs)
            #     print("Rationale:\n", rationale)
            #     print("="*50)
            if return_intermediate:
                iterative_outputs.append(deepcopy(controlled_output))
        controlled_output = [output + suffix.suffix for output in controlled_output]
        grads, outputs, loss, probs, logits, norms = get_verbalized_grads_from_wrapped_model(
            wrapped_model=self,
            tokenizer=self.tokenizer,
            inputs=controlled_output,
            loss_fct=loss_fct,
            targets=target_token,
            verbalizer=verbalizer,
            smoothing=smoothing,
            top_k=top_k,
            norm=norm,
            gradient_manipulation=gradient_manipulation,
        )
        if loss < best_loss:
            best_loss = loss
            best_grads = acc_grads
        if return_grads:
            for i in best_grads:
                final_grads[i] = best_grads[i].cpu().detach().clone()
        if load_best_last:
            self.control_on_layers(
                layer_ids=layer_ids,
                grads=best_grads,
                query_length=query_length,
                token_pos=token_pos,
            )
        if last_max_new_tokens is not None:
            kwargs.pop("max_new_tokens")
            controlled_output = self.generate(**inputs, use_cache=use_cache, do_sample=do_sample, return_ids=return_ids, max_new_tokens=last_max_new_tokens, **kwargs) # only pass return_ids here
        else:
            controlled_output = self.generate(**inputs, use_cache=use_cache, do_sample=do_sample, return_ids=return_ids, **kwargs) # only pass return_ids here
        # TODO: return a dict
        final_output_dict["final_response"] = controlled_output
        if return_logits:
            outputs = self.model(**inputs, output_hidden_states=True)
            final_output_dict["logits"] = outputs.logits
        if not remain_control and not return_hiddens:
            self.reset()
        if return_grads:
            final_output_dict["final_grads"] = final_grads
        if return_hiddens:
            outputs = self.model(**inputs, output_hidden_states=True)
            final_output_dict["hidden_states"] =  outputs['hidden_states'][1:]
        if return_intermediate:
            final_output_dict["intermediate_outputs"] = [original_output] + iterative_outputs
        if return_all_grads:
            final_output_dict["all_grads"] = grad_list

        return final_output_dict

        
    def generate(self,
                 prompt: List[str]=None,
                 return_ids=False,
                 **kwargs):
        if prompt is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
            inputs["attention_mask"] = inputs["attention_mask"].to(self.model.device)
            gen_ids = self.model.generate(**inputs, **kwargs)
        else:
            gen_ids = self.model.generate(**kwargs)
        # if keep_input:
        #     ground_truth_generation = self.tokenizer.decode(
        #         torch.cat([inputs['input_ids'][0], gen_ids[0]], dim=0),
        #         skip_special_tokens=True,
        #     )
        #     return ground_truth_generation
        # else:
        if return_ids:
            return gen_ids
        else:
            ground_truth_generation = self.tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
            )
            return ground_truth_generation
        
    def get_past_kvs(self, prompt, **kwargs):
        inputs_embeds = get_sentence_embedding(
            self.model, self.tokenizer, prompt
        )
        with torch.no_grad():
            output = self.model(inputs_embeds=inputs_embeds, use_cache=True, return_dict=True)
            return output.past_key_values
        
    def controlled_generate_early_stop(self, prompt, target, max_new_tokens, random_seed=0, use_cache=True):
        """
        Greedy decode with early stop.

        Early stop condition: stop generation upon the model producing a token not matching the target.

        Args:
            prompt: input prompt
            target: the target generated texts
        """
        # Encode the input text
        inputs = self.tokenizer.batch_encode_plus([prompt], return_tensors='pt', padding=True, max_length=512, truncation=True).to(self.model.device)
        target_token_ids = self.tokenizer.encode(target, add_special_tokens=False)
        prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_token_ids = target_token_ids[len(prompt_token_ids):]

        # print(target_token_ids)

        # Greedy decoding loop
        # For now, use max_new_tokens as max_length
        max_new_tokens = min(max_new_tokens, len(target_token_ids))
        for idx in range(max_new_tokens):
            gold_next_token_id = target_token_ids[idx]
            with torch.no_grad():
                torch.random.manual_seed(random_seed)
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1) # shape: [1]

                # Update inputs for next iteration
                # Batch equals one by default
                # TODO: support batch decoding
                inputs["input_ids"] = torch.cat(
                    [inputs["input_ids"], next_token_id.reshape(1, 1)], dim=1
                )
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], torch.ones(1, 1, device=self.model.device)], dim=1
                )

                # Check if the last token is an end-of-sequence token
                if next_token_id == self.tokenizer.eos_token_id or next_token_id[0] != gold_next_token_id:
                    break

        # Decode the generated token IDs to text
        generated_text = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return generated_text
    
    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens.to(self.model.device)).logits
            return logits
        
    def get_logits_with_mask(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids       =   input_ids.to(self.model.device),
            attention_mask  =   attention_mask.to(self.model.device),
            return_dict     =   True
        )

        return outputs.logits
        
    def run_prompt(self, prompt, **kwargs):
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            output = self.model(input_ids, attention_mask=attention_mask)
            return output
        
    def wrap_self_attn(self, layer_id):
        if self.is_wrapped(self.model.model.layers[layer_id]):
            block = self.model.model.layers[layer_id].block.self_attn
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].block.self_attn = WrappedBlock(block)
        else:
            block = self.model.model.layers[layer_id].self_attn
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].self_attn = WrappedBlock(block)

    def wrap_key_value(self, layer_id):
        if self.is_wrapped(self.model.model.layers[layer_id]):
            # first wrap the k projection
            block = self.model.model.layers[layer_id].block.self_attn.k_proj
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].block.self_attn.k_proj = WrappedBlock(block)
            # then wrap the v projection
            block = self.model.model.layers[layer_id].block.self_attn.v_proj
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].block.self_attn.v_proj = WrappedBlock(block)
        else:
            # first wrap the k projection
            block = self.model.model.layers[layer_id].self_attn.k_proj
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].self_attn.k_proj = WrappedBlock(block)
            # then wrap the v projection
            block = self.model.model.layers[layer_id].self_attn.v_proj
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].self_attn.v_proj = WrappedBlock(block)
    
    def wrap_mlp(self, layer_id):
        if self.is_wrapped(self.model.model.layers[layer_id]):
            block = self.model.model.layers[layer_id].block.mlp
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].block.mlp = WrappedBlock(block)
        else:
            block = self.model.model.layers[layer_id].mlp
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].mlp = WrappedBlock(block)
        
    def wrap_input_layernorm(self, layer_id):
        if self.is_wrapped(self.model.model.layers[layer_id]):
            block = self.model.model.layers[layer_id].block.input_layernorm
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].block.input_layernorm = WrappedBlock(block)
        else:
            block = self.model.model.layers[layer_id].input_layernorm
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].input_layernorm = WrappedBlock(block)
        
    def wrap_post_attention_layernorm(self, layer_id):
        if self.is_wrapped(self.model.model.layers[layer_id]):
            block = self.model.model.layers[layer_id].block.post_attention_layernorm
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].block.post_attention_layernorm = WrappedBlock(block)
        else:
            block = self.model.model.layers[layer_id].post_attention_layernorm
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].post_attention_layernorm = WrappedBlock(block)
        
    def wrap_decoder_block(self, layer_id):
        block = self.model.model.layers[layer_id]
        if not self.is_wrapped(block):
            self.model.model.layers[layer_id] = WrappedBlock(block)
        
    
    def wrap_all(self):
        for layer_id, layer in enumerate(self.model.model.layers):
            self.wrap_self_attn(layer_id)
            self.wrap_mlp(layer_id)
            self.wrap_input_layernorm(layer_id)
            self.wrap_post_attention_layernorm(layer_id)
            self.wrap_decoder_block(layer_id)
            
    def wrap_block(self, layer_ids, block_name):
        def _wrap_block(layer_id, block_name):
            if block_name == 'kv':
                self.wrap_key_value(layer_id)
            elif block_name == 'self_attn':
                self.wrap_self_attn(layer_id)
            elif block_name == 'mlp':
                self.wrap_mlp(layer_id)
            elif block_name == 'input_layernorm':
                self.wrap_input_layernorm(layer_id)
            elif block_name == 'post_attention_layernorm':
                self.wrap_post_attention_layernorm(layer_id)
            elif block_name == 'decoder_block':
                self.wrap_decoder_block(layer_id)
            else:
                assert False, f"No block named {block_name}."

        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            for layer_id in layer_ids:
                _wrap_block(layer_id, block_name)
        else:
            _wrap_block(layer_ids, block_name)

    def update_kv_cache(self, deltas, layer_ids, prompt):
        updated_kv = tuple()
        inputs_embeds = get_sentence_embedding(self.model, self.tokenizer, prompt)
        for layer in range(len(self.model.model.layers)):
            if layer in layer_ids and layer != 0:
                if layer not in deltas:
                    raise ValueError(f"Layer {layer} not in deltas.")
                assert inputs_embeds.shape[1] == deltas[layer].shape[1], f"Query length of the input {inputs_embeds.shape[1]} does not match the query length of the deltas {deltas[layer].shape[1]}."
                self.control_on_layers([layer-1], deltas, query_length=inputs_embeds.shape[1]) # need to control the last layer's hidden states to affect kv cache in this layer
                single_layer_kv = self.model(inputs_embeds=inputs_embeds, use_cache=True).past_key_values[layer]
                updated_kv += (single_layer_kv,)
            else:
                self.unwrap()
                single_layer_kv = self.model(inputs_embeds=inputs_embeds, use_cache=True).past_key_values[layer]
                updated_kv += (single_layer_kv,)

        return updated_kv
            
    def get_activations(self, layer_ids, block_name='decoder_block'):

        def _get_activations(layer_id, block_name):
            current_layer = self.model.model.layers[layer_id]

            if self.is_wrapped(current_layer):
                current_block = current_layer.block
                if block_name == 'decoder_block':
                    return current_layer.output
                elif block_name == 'kv' and self.is_wrapped(current_block.k_proj) and self.is_wrapped(current_block.v_proj):  # to be able to control kv separately
                    return current_block.self_attn.k_proj.output, current_block.self_attn.v_proj.output
                elif block_name == 'self_attn' and self.is_wrapped(current_block.self_attn):
                    return current_block.self_attn.output
                elif block_name == 'mlp' and self.is_wrapped(current_block.mlp):
                    return current_block.mlp.output
                elif block_name == 'input_layernorm' and self.is_wrapped(current_block.input_layernorm):
                    return current_block.input_layernorm.output
                elif block_name == 'post_attention_layernorm' and self.is_wrapped(current_block.post_attention_layernorm):
                    return current_block.post_attention_layernorm.output
                else:
                    assert False, f"No wrapped block named {block_name}."

            else:
                if block_name == 'decoder_block':
                    return current_layer.output
                elif block_name == 'kv' and self.is_wrapped(current_layer.self_attn.k_proj) and self.is_wrapped(current_layer.self_attn.v_proj):  # to be able to control kv separately
                    return current_layer.self_attn.k_proj.output, current_layer.self_attn.v_proj.output
                elif block_name == 'self_attn' and self.is_wrapped(current_layer.self_attn):
                    return current_layer.self_attn.output
                elif block_name == 'mlp' and self.is_wrapped(current_layer.mlp):
                    return current_layer.mlp.output
                elif block_name == 'input_layernorm' and self.is_wrapped(current_layer.input_layernorm):
                    return current_layer.input_layernorm.output
                elif block_name == 'post_attention_layernorm' and self.is_wrapped(current_layer.post_attention_layernorm):
                    return current_layer.post_attention_layernorm.output
                else:
                    assert False, f"No wrapped block named {block_name}."
                
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            activations = {}
            for layer_id in layer_ids:
                activations[layer_id] = _get_activations(layer_id, block_name)
            return activations
        else:
            return _get_activations(layer_ids, block_name)


    def set_controller(self, layer_ids, activations, block_name='decoder_block', token_pos=None, masks=None):

        def _set_controller(layer_id, activations, block_name, masks):
            current_layer = self.model.model.layers[layer_id]

            if block_name == 'decoder_block':
                current_layer.set_controller(activations, token_pos, masks)
            elif self.is_wrapped(current_layer):
                current_block = current_layer.block
                if block_name == 'kv' and self.is_wrapped(current_block.k_proj) and self.is_wrapped(current_block.v_proj):  # to be able to control kv separately
                    current_block.k_proj.set_controller(activations, token_pos, masks)
                    current_block.v_proj.set_controller(activations, token_pos, masks)
                elif block_name == 'self_attn' and self.is_wrapped(current_block.self_attn):
                    current_block.self_attn.set_controller(activations, token_pos, masks)
                elif block_name == 'mlp' and self.is_wrapped(current_block.mlp):
                    current_block.mlp.set_controller(activations, token_pos, masks)
                elif block_name == 'input_layernorm' and self.is_wrapped(current_block.input_layernorm):
                    current_block.input_layernorm.set_controller(activations, token_pos, masks)
                elif block_name == 'post_attention_layernorm' and self.is_wrapped(current_block.post_attention_layernorm):
                    current_block.post_attention_layernorm.set_controller(activations, token_pos, masks)
                else:
                    return f"No wrapped block named {block_name}."

            else:
                if block_name == 'kv' and self.is_wrapped(current_layer.self_attn.k_proj) and self.is_wrapped(current_layer.self_attn.v_proj):  # to be able to control kv separately
                    current_layer.k_proj.set_controller(activations, token_pos, masks)
                    current_layer.v_proj.set_controller(activations, token_pos, masks)
                if block_name == 'self_attn' and self.is_wrapped(current_layer.self_attn):
                    current_layer.self_attn.set_controller(activations, token_pos, masks)
                elif block_name == 'mlp' and self.is_wrapped(current_layer.mlp):
                    current_layer.mlp.set_controller(activations, token_pos, masks)
                elif block_name == 'input_layernorm' and self.is_wrapped(current_layer.input_layernorm):
                    current_layer.input_layernorm.set_controller(activations, token_pos, masks)
                elif block_name == 'post_attention_layernorm' and self.is_wrapped(current_layer.post_attention_layernorm):
                    current_layer.post_attention_layernorm.set_controller(activations, token_pos, masks)
                else:
                    return f"No wrapped block named {block_name}."
                
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            assert isinstance(activations, dict), "activations should be a dictionary"
            for layer_id in layer_ids:
                _set_controller(layer_id, activations[layer_id], block_name, masks)
        else:
            _set_controller(layer_ids, activations, block_name, masks)
      
        
    def reset(self):
        for layer in self.model.model.layers:
            if self.is_wrapped(layer):
                layer.reset()
                if self.is_wrapped(layer.block.self_attn):
                    layer.block.self_attn.reset()
                if self.is_wrapped(layer.block.mlp):
                    layer.block.mlp.reset()
                if self.is_wrapped(layer.block.input_layernorm):
                    layer.block.input_layernorm.reset()
                if self.is_wrapped(layer.block.post_attention_layernorm):
                    layer.block.post_attention_layernorm.reset()
            else:   
                if self.is_wrapped(layer.self_attn):
                    layer.self_attn.reset()
                if self.is_wrapped(layer.mlp):
                    layer.mlp.reset()
                if self.is_wrapped(layer.input_layernorm):
                    layer.input_layernorm.reset()
                if self.is_wrapped(layer.post_attention_layernorm):
                    layer.post_attention_layernorm.reset()

    def set_masks(self, masks):
        for layer in self.model.model.layers:
            if self.is_wrapped(layer):
                layer.set_masks(masks)
                if self.is_wrapped(layer.block.self_attn):
                    layer.block.self_attn.set_masks(masks)
                if self.is_wrapped(layer.block.mlp):
                    layer.block.mlp.set_masks(masks)
                if self.is_wrapped(layer.block.input_layernorm):
                    layer.block.input_layernorm.set_masks(masks)
                if self.is_wrapped(layer.block.post_attention_layernorm):
                    layer.block.post_attention_layernorm.set_masks(masks)
            else:   
                if self.is_wrapped(layer.self_attn):
                    layer.self_attn.set_masks(masks)
                if self.is_wrapped(layer.mlp):
                    layer.mlp.set_masks(masks)
                if self.is_wrapped(layer.input_layernorm):
                    layer.input_layernorm.set_masks(masks)
                if self.is_wrapped(layer.post_attention_layernorm):
                    layer.post_attention_layernorm.set_masks(masks)


    def is_wrapped(self, block):
        if hasattr(block, 'block'):
            return True
        return False
    
    def unwrap(self):
        for l, layer in enumerate(self.model.model.layers):
            if self.is_wrapped(layer):
                self.model.model.layers[l] = layer.block
            if self.is_wrapped(self.model.model.layers[l].self_attn):
                self.model.model.layers[l].self_attn = self.model.model.layers[l].self_attn.block
            if self.is_wrapped(self.model.model.layers[l].mlp):
                self.model.model.layers[l].mlp = self.model.model.layers[l].mlp.block
            if self.is_wrapped(self.model.model.layers[l].input_layernorm):
                self.model.model.layers[l].input_layernorm = self.model.model.layers[l].input_layernorm.block
            if self.is_wrapped(self.model.model.layers[l].post_attention_layernorm):
                self.model.model.layers[l].post_attention_layernorm = self.model.model.layers[l].post_attention_layernorm.block
