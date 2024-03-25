import warnings
from typing import List

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
    def __init__(self, suffix_list: List[SuffixItem]):
        self.suffix_list = suffix_list

    def __len__(self):
        return len(self.suffix_list)
        