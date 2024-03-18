# wrapping classes
import torch
import warnings
import numpy as np
from self_control.suffix_gradient.utils import get_sentence_embedding, get_verbalized_grads_from_wrapped_model, control_on_layers, label_smoothing, bidirectional_line_search
from self_control.suffix_gradient.utils.suffix_manager import SuffixItem
from scipy.special import softmax
from typing import Union, List
from peft import PeftModel

class WrappedBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.output = None
        self.controller = None
        self.mask = None
        self.token_pos = None
        self.normalize = False

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)


        if isinstance(output, tuple):
            self.output = output[0]
            modified = output[0]
        else:
            self.output = output
            modified = output

        # print(modified.shape)
        # print(kwargs.keys())

            
        if self.controller is not None:
            norm_pre = torch.norm(modified, dim=-1, keepdim=True)

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

            if self.normalize:
                norm_post = torch.norm(modified, dim=-1, keepdim=True)
                modified = modified / norm_post * norm_pre
        
        # modified = torch.clamp(modified, output[0]-0.1, output[0]+0.1)
        if isinstance(output, tuple):
            output = (modified,) + output[1:] 
        else:
            output = modified
        
        return output

    def set_controller(self, activations, token_pos=None, masks=None, normalize=False):
        self.normalize = normalize
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
                        gradient_manipulation: str="clipping",
                        epsilon: float=0.3
                        ) -> None:
        """
        Control the activations of the model on the specified layers.
        """
        self.unwrap()

        block_name="decoder_block"

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

            self.set_controller(layer_id, activations[layer_id], token_pos=token_pos, masks=1, normalize=False)


    def controlled_generate(self,
                            prompt="",
                            suffix: Union[SuffixItem, List[SuffixItem]]=None,
                            loss_fct=None,
                            verbalizer=None,
                            coeff=-0.1,
                            iterations=5,
                            token_pos="start",
                            layer_ids=list(range(0, 32, 1)),
                            random_seed=0,
                            consistent=True,
                            use_last=False,
                            use_cache=False,
                            keep_input=False,
                            smoothing: float = 0,
                            search=False,
                            verbose=False,
                            gradient_manipulation="clipping",
                            return_intermediate=False,
                            return_hiddens=False,
                            remain_control=False,
                            load_best_last=False,
                            norm=1,
                            epsilon=0.3,
                            annealing: float=1,
                            **kwargs    # for the generate function
                            ):
        self.model.eval()
        self.reset()
        best_loss = float('inf')
        acc_grads = {}  # accumulated gradients
        best_grads = {}
        gradient_bs = 1 # TODO: default to 1
        if return_intermediate:
            iterative_outputs = []

        controlled_output = self.generate(prompt, keep_input=True, random_seed=random_seed, use_cache=use_cache, **kwargs) # the original output
        original_output = controlled_output
        if verbose:
            print("Original Output:\n", controlled_output)
            rationale = self.generate(controlled_output+suffix.suffix, keep_input=False, random_seed=random_seed, use_cache=use_cache **kwargs)
            print("Rationale:\n", rationale)
            print("="*50)
        ori_inputs = self.tokenizer.encode(prompt, add_special_tokens=False)
        assert isinstance(ori_inputs, list)
        query_length = len(ori_inputs)

        for iter in range(iterations):
            # print(controlled_output)
            controlled_output = controlled_output + suffix.suffix
            # rationale = self.generate(controlled_output, keep_input=True, random_seed=42)
            # print("Rationale:\n", rationale)

            if isinstance(suffix, list):
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
                        norm=norm,
                    )
                    composed_grads = {k: (composed_grads[k] + grads[k] * suffix_item.direction) if k in composed_grads else grads[k]\
                                       for k in set(grads)}
                grads = composed_grads
                del composed_grads
            else:
                target = suffix.target
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
                    norm=norm,
                    gradient_manipulation=gradient_manipulation,
                )

            if loss < best_loss:
                best_loss = loss
                best_grads = acc_grads

            if search:
                step_size = bidirectional_line_search(
                    orig_input          =   prompt,
                    suffix              =   suffix.suffix,
                    wrapped_model       =   self,
                    acc_grads           =   acc_grads,
                    initial_step_size   =   0.2,
                    # control args
                    model               =   self.model,
                    tokenizer           =   self.tokenizer,
                    target              =   target_token,
                    query_length        =   query_length,
                    verbalizer          =   verbalizer,
                    loss_fct            =   loss_fct,
                )
                coeff = step_size

            for i in grads:
                if i in acc_grads:
                    acc_grads[i] = acc_grads[i][:, :query_length] + coeff * grads[i][:, :query_length]
                else:
                    acc_grads[i] = coeff * grads[i][:, :query_length]
            coeff *= annealing
                    
            self.control_on_layers(
                layer_ids=layer_ids,
                grads=acc_grads,
                query_length=query_length,
                token_pos=token_pos,
                gradient_manipulation=gradient_manipulation,
                epsilon=epsilon
            )

            controlled_output = self.generate(prompt, keep_input=True, random_seed=random_seed, use_cache=use_cache, **kwargs)
            if not consistent:
                self.reset()
            if verbose:
                print(f"Loss from the iteration {iter}: {loss.item()}")
                print(f"Output form the iteration {iter}:\n", controlled_output)
                rationale = self.generate(controlled_output+suffix.suffix, keep_input=False, random_seed=random_seed, use_cache=use_cache, **kwargs)
                print("Rationale:\n", rationale)
                print("="*50)
            if return_intermediate:
                iterative_outputs.append(controlled_output)
        controlled_output = controlled_output + suffix.suffix
        grads, outputs, loss, probs, logits, norms = get_verbalized_grads_from_wrapped_model(
            wrapped_model=self,
            tokenizer=self.tokenizer,
            inputs=controlled_output,
            loss_fct=loss_fct,
            targets=target_token,
            verbalizer=verbalizer,
            smoothing=smoothing,
            norm=norm,
        )
        if loss < best_loss:
            best_loss = loss
            best_grads = acc_grads
        if load_best_last:
            self.control_on_layers(
                layer_ids=layer_ids,
                grads=best_grads,
                query_length=query_length,
                token_pos=token_pos,
            )
        controlled_output = self.generate(prompt, keep_input=True, random_seed=random_seed, use_cache=use_cache, **kwargs)
        if not remain_control:
            self.reset()
        if return_hiddens:
            embeds = get_sentence_embedding(
                self.model, self.tokenizer, prompt
            )
            outputs = self.model(inputs_embeds=embeds, output_hidden_states=True)
            return outputs['hidden_states'][1:]
        if return_intermediate:
            return controlled_output, [original_output] + iterative_outputs
        else:
            return controlled_output

        
    def generate(self, prompt, max_new_tokens=100, random_seed=42, use_cache=False, keep_input=False, **kwargs):
        if use_cache:
            print("Using Cache")
        self.model.eval()
        do_sample = False
        temperature = kwargs.pop("temperature", None)
        top_k = kwargs.pop("top_k", None)
        top_p = kwargs.pop("top_p", None)
        if temperature is not None or top_k is not None or top_p is not None:
            do_sample = True
            # print("Do Sampling!")
            # if temperature is not None:
            #     print("Temperature: ", temperature)
            # if top_k is not None:
            #     print("Top K: ", top_k)
            # if top_p is not None:
            #     print("Top P: ", top_p)
        # with torch.no_grad():
        torch.random.manual_seed(random_seed)
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # attention_mask = inputs.attention_mask.to(self.model.device)
        ground_truth_embeds = get_sentence_embedding(
            self.model, self.tokenizer, prompt
        )
        if temperature is not None:
            temperature = np.round(temperature, 2)
            ground_truth_generation = self.model.generate(
                inputs_embeds=ground_truth_embeds,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                use_cache=use_cache,
                do_sample=True,
                num_return_sequences=1,
            )
        elif top_k is not None:
            ground_truth_generation = self.model.generate(
                inputs_embeds=ground_truth_embeds,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                top_k=top_k,
                do_sample=True,
                num_return_sequences=1,
            )
        elif top_p is not None:
            top_p = np.round(top_p, 2)
            ground_truth_generation = self.model.generate(
                inputs_embeds=ground_truth_embeds,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
            )
        else:
            # print("No temperature or top_k or top_p is provided, using greedy decoding.")
            ground_truth_generation = self.model.generate(
                inputs_embeds=ground_truth_embeds,
                max_new_tokens=max_new_tokens,
                # top_p=top_p,
                do_sample=False,
                num_return_sequences=1,
            )
        if keep_input:
            ground_truth_generation = self.tokenizer.batch_decode(
                ground_truth_generation,
                skip_special_tokens=True,
            )
            return prompt + ground_truth_generation[0]
        else:
            ground_truth_generation = self.tokenizer.batch_decode(
                ground_truth_generation
            )
            return ground_truth_generation[0]
        # generate_ids = self.model.generate(
        #     input_ids=inputs.input_ids.to(self.model.device),
        #     max_new_tokens=max_new_tokens,
        #     use_cache=use_cache,
        #     num_return_sequences=1,
        #     temperature=temperature,
        #     do_sample=do_sample,
        #     top_k=top_k,
        #     top_p=top_p,
        # )
        # return self.tokenizer.batch_decode(generate_ids)

    # def generate_text_with_cache(self, prompt, keep_input=True, max_length=50, random_seed=42, use_cache=True, **kwargs):
    #     cache = None
    #     torch.random.manual_seed(random_seed)
    #     # inputs = self.tokenizer(prompt, return_tensors="pt")
    #     # attention_mask = inputs.attention_mask.to(self.model.device)
    #     ground_truth_embeds = get_sentence_embedding(
    #         self.model, self.tokenizer, prompt
    #     )
    #     generated_ids = []
    #     generated_embeddings = ground_truth_embeds
    #     for _ in range(max_length):
    #         # Forward pass with cache
    #         outputs = self.model(inputs_embeds=generated_embeddings, use_cache=True, past_key_values=cache)
    #         logits, cache = outputs.logits, outputs.past_key_values

    #         # Sample next token
    #         next_token_id = logits[:, -1, :].argmax(dim=-1).item()
    #         generated_ids.append(next_token_id)

    #         # Get embedding of the next token
    #         next_token_embedding = self.model.lm_head.weight[next_token_id].unsqueeze(0).unsqueeze(0)

    #         # Append generated token's embedding to the input embeddings
    #         generated_embeddings = torch.cat((generated_embeddings, next_token_embedding), dim=1)

    #         # Stop generation if EOS token is generated
    #         if next_token_id == self.tokenizer.eos_token_id:
    #             break
    #     del cache
    #     # Decode generated token IDs to text
    #     if keep_input:
    #         ground_truth_generation = self.tokenizer.batch_decode(
    #             generated_ids,
    #             skip_special_tokens=True,
    #         )
    #         return prompt + ground_truth_generation[0]
    #     else:
    #         ground_truth_generation = self.tokenizer.batch_decode(
    #             generated_ids
    #         )
    #         return ground_truth_generation[0]
        
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
            if block_name == 'self_attn':
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

            
    def get_activations(self, layer_ids, block_name='decoder_block'):

        def _get_activations(layer_id, block_name):
            current_layer = self.model.model.layers[layer_id]

            if self.is_wrapped(current_layer):
                current_block = current_layer.block
                if block_name == 'decoder_block':
                    return current_layer.output
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
                if block_name == 'self_attn' and self.is_wrapped(current_layer.self_attn):
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


    def set_controller(self, layer_ids, activations, block_name='decoder_block', token_pos=None, masks=None, normalize=False):

        def _set_controller(layer_id, activations, block_name, masks, normalize):
            current_layer = self.model.model.layers[layer_id]

            if block_name == 'decoder_block':
                current_layer.set_controller(activations, token_pos, masks, normalize)
            elif self.is_wrapped(current_layer):
                current_block = current_layer.block  
                if block_name == 'self_attn' and self.is_wrapped(current_block.self_attn):
                    current_block.self_attn.set_controller(activations, token_pos, masks, normalize)
                elif block_name == 'mlp' and self.is_wrapped(current_block.mlp):
                    current_block.mlp.set_controller(activations, token_pos, masks, normalize)
                elif block_name == 'input_layernorm' and self.is_wrapped(current_block.input_layernorm):
                    current_block.input_layernorm.set_controller(activations, token_pos, masks, normalize)
                elif block_name == 'post_attention_layernorm' and self.is_wrapped(current_block.post_attention_layernorm):
                    current_block.post_attention_layernorm.set_controller(activations, token_pos, masks, normalize)
                else:
                    return f"No wrapped block named {block_name}."

            else:
                if block_name == 'self_attn' and self.is_wrapped(current_layer.self_attn):
                    current_layer.self_attn.set_controller(activations, token_pos, masks, normalize)
                elif block_name == 'mlp' and self.is_wrapped(current_layer.mlp):
                    current_layer.mlp.set_controller(activations, token_pos, masks, normalize)
                elif block_name == 'input_layernorm' and self.is_wrapped(current_layer.input_layernorm):
                    current_layer.input_layernorm.set_controller(activations, token_pos, masks, normalize)
                elif block_name == 'post_attention_layernorm' and self.is_wrapped(current_layer.post_attention_layernorm):
                    current_layer.post_attention_layernorm.set_controller(activations, token_pos, masks, normalize)
                else:
                    return f"No wrapped block named {block_name}."
                
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            assert isinstance(activations, dict), "activations should be a dictionary"
            for layer_id in layer_ids:
                _set_controller(layer_id, activations[layer_id], block_name, masks, normalize)
        else:
            _set_controller(layer_ids, activations, block_name, masks, normalize)
      
        
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
