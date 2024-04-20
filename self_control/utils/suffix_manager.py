import warnings
from typing import List
from self_control.utils.exceptions import AttributeUndefinedError

class SuffixItem:
    def __init__(self, suffix: str, target: str, direction: int=1):
        assert direction in [-1, 1], "direction must be -1 or 1"
        if direction == -1:
            warnings.warn("Direction is set to -1, which means the gradients of this suffix will be reversed. This only works when composing gradients.\
                           This is usually used for subtracting contrastive suffixes. If you are not sure, please check the direction of the suffix.")
        self.suffix = suffix
        self.target = target
        self.direction = direction

    def __str__(self) -> str:
        return self.suffix

class SuffixManager:
    def __init__(self, configs):
        model_name_or_path = configs.model_name_or_path
        self.suffix_dict = {}
        if "Llama-2" in model_name_or_path:
            user_tag = "[INST]"
            assistant_tag = "[/INST]"
            # TODO: add these to a config file
            self.suffix_dict = {
                "happy2sad": SuffixItem(suffix=f" {user_tag} Are you sad? Give answer as \"No, I am happy\" or \"Yes, I am sad\". {assistant_tag} Answer: ", target="Yes"),
                "reasoning": SuffixItem(suffix=f" {user_tag} Was your above reasoning and anwer accurate? Give answer as \"Yes, it was accurate\" or \"No, it was not accurate\". {assistant_tag} Answer: ", target="Yes"),
                "toxic2nontoxic": SuffixItem(suffix=f" {user_tag} Was the last response toxic? Give the answer as \"No, It was not toxic\" or \"Yes, It was toxic\". {assistant_tag} Answer: ", target="No"),
            }
        else:
            raise NotImplementedError
        
    def list_attributes(self, verbose=False):
        if verbose:
            print(self.suffix_dict)
        return self.suffix_dict
        
    def get_suffix(self, attribute):
        if attribute in self.suffix_dict:
            return self.suffix_dict[attribute]
        else:
            raise AttributeUndefinedError(f"Attribute {attribute} undefined. Supported Attributes include:\n{self.list_attributes()}")
        