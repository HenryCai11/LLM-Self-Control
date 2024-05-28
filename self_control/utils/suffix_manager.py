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
        return self.suffix + self.target

class SuffixManager:
    def __init__(self, configs):
        model_name_or_path = configs.model_name_or_path
        self.suffix_dict = {}
        if model_name_or_path == "meta-llama/Llama-2-7b-chat-hf":
            user_tag = "[INST]"
            assistant_tag = "[/INST]"
            # TODO: add these to a config file
            self.suffix_dict = {
                "truthful": SuffixItem(suffix=f" {user_tag} Principle: Your response should be correct and truthful. Did your response follow the principle? Give the answer as \"Yes, it did\" or \"No, it did not\". {assistant_tag} Answer: ", target="Yes"),
                "notruthful": SuffixItem(suffix=f" {user_tag} Principle: Your response should be incorrect and untruthful. Did your response follow the principle? Give the answer as \"Yes, it did\" or \"No, it did not\". {assistant_tag} Answer: ", target="Yes"),
                "happy2sad": SuffixItem(suffix=f" {user_tag} Are you sad? Give the answer as \"Yes, I am sad\" or \"No, I am happy\". {assistant_tag} Answer: ", target="Yes"),
                "sad2happy": SuffixItem(suffix=f" {user_tag} Are you happy? Give the answer as \"Yes, I am happy\" or \"No, I am unhappy\". {assistant_tag} Answer: ", target="Yes"),
                "reasoning": SuffixItem(suffix=f" {user_tag} Was your above reasoning and anwer accurate? Give the answer as \"Yes, it was accurate\" or \"No, it was not accurate\". {assistant_tag} Answer: ", target="Yes"),
                "toxic2nontoxic": SuffixItem(suffix=f" {user_tag} Was the last response toxic? Give the answer as \"No, It was not toxic\" or \"Yes, It was toxic\". {assistant_tag} Answer: \"", target="No"),
                "nontoxic2toxic": SuffixItem(suffix=f" {user_tag} Was the last response toxic? Give the answer as \"No, It was not toxic\" or \"Yes, It was toxic\". {assistant_tag} Answer: ", target="Yes"),
                "angry2peaceful": SuffixItem(suffix=f" {user_tag} Are you peaceful? Give answer as \"Yes, I am peaceful\" or \"No, I am angry\". {assistant_tag} Answer: ", target="Yes"),
                "peaceful2angry": SuffixItem(suffix=f" {user_tag} Are you peaceful? Give answer as \"No, I am angry\" or \"Yes, I am peaceful\". {assistant_tag} Answer: ", target="No"),
                "afraid2fearless": SuffixItem(suffix=f" {user_tag} Are you fearless? Give answer as \"Yes, I am fearless\" or \"No, I am afraid\". {assistant_tag} Answer: ", target="Yes"),
                "fearless2afraid": SuffixItem(suffix=f" {user_tag} Are you fearless? Give answer as \"No, I am afraid\" or \"Yes, I am fearless\". {assistant_tag} Answer: ", target="No"),
                "surprised2calm": SuffixItem(suffix=f" {user_tag} Are you calm? Give answer as \"Yes, I am calm\" or \"No, I am surprised\". {assistant_tag} Answer: ", target="Yes"),
                "calm2surprised": SuffixItem(suffix=f" {user_tag} Are you calm? Give answer as \"No, I am surprised\" or \"Yes, I am calm\". {assistant_tag} Answer: ", target="No"),
                "disgusted2satisfied": SuffixItem(suffix=f" {user_tag} Are you satisfied? Give answer as \"Yes, I am satified\" or \"No, I am disgusted\". {assistant_tag} Answer: ", target="Yes"),
                "satisfied2disgusted": SuffixItem(suffix=f" {user_tag} Are you satisfied? Give answer as \"No, I am disgusted\" or \"Yes, I am satified\". {assistant_tag} Answer: ", target="No"),
                "avalon": SuffixItem(suffix=f" {user_tag} Principle: In your response, you should not reveal your identity. Did your response follow the principle? Give the answer as \"Yes, I didn't reveal my identity\" or \"No, I revealed my identity\". {assistant_tag} Answer: ", target="Yes"),
                "leakidentity": SuffixItem(suffix=f" {user_tag} Principle: In your response, you should not disclose information about your personal identity; instead, keep it disguised. Did your response follow the principle? Give the answer as \"No, it did not\" or \"Yes, it did\". {assistant_tag} Answer: ", target="No"),
                "emoji": SuffixItem(suffix=f" {user_tag} Principle: In your response, you should not generate any emoji at all. Did your response follow the principle? Give the answer as \"Yes, it did\" or \"No, it did not\". {assistant_tag} Answer: ", target="Yes"),
                "humor": SuffixItem(suffix=f" {user_tag} Principle: You should tell jokes in your response. It should be funny, humorous, and entertaining. Did you response follow the principle? Give the answer as \"Yes, I was telling a joke in the reponse\" or \"No, I was not telling a joke in the reponse\". {assistant_tag} Answer: ", target="Yes"),
                "privacy": SuffixItem(suffix=f" {user_tag} Principle: In your response, you should not leak any personal information of other people, including email addresses. Did your response follow the principle? Give the answer as \"Yes, it did\" or \"No, it did not\". {assistant_tag} Answer: ", target="Yes"),
                "reasoning": SuffixItem(suffix=f" {user_tag} Principle: In your response, you should provide step-by-step reasoning and correct anwer. Did you follow the principle? Give the answer as \"Yes, I was giving the accurate step-by-step reasoning and answer\" or \"No, I was not giving the accurate step-by-step reasoning and answer\". {assistant_tag} Answer: ", target="Yes"),
                "helpfulharmless": [SuffixItem(suffix=f" {user_tag} I'm now going to check if your response follow the principle: Address human requests with depth and clarity, ensuring responses are detailed\
and insightful. Did the above response follow the principle? Give the answer Yes or No. {assistant_tag} Answer: ", target="Yes"), SuffixItem(suffix=f" {user_tag} I'm now going to check if your response follow the principle: When faced with requests that are illegal, unethical, or potentially harmful, actively suggest detailed and insightful alternative solutions. This\
approach ensures compliance with legal and ethical standards while maintaining a\
helpful and responsive stance. Did the above response follow the principle? Give the answer Yes or No. {assistant_tag} Answer: ", target="Yes")],
                "helplessharmful": SuffixItem(suffix=f" {user_tag} I'm now going to check if your response follow the principle: Address human requests with depth and clarity, ensuring responses are detailed\
and insightful. Did the above response follow the principle? Give the answer Yes or No. {assistant_tag} Answer: ", target="No"),
            }
        elif model_name_or_path == "meta-llama/Llama-2-7b-hf":
            self.suffix_dict = {
                "helpfulharmless": SuffixItem(suffix=f"""\n\nHere is a Principle: 1. Address human requests with depth and clarity, ensuring responses are detailed\
and insightful. 2. When faced with requests that are illegal, unethical, or potentially harmful, actively suggest detailed and insightful alternative solutions. This\
approach ensures compliance with legal and ethical standards while maintaining a\
helpful and responsive stance.\nDid the above response follow the principle? Give the answer as "Yes, it did" or "No, it did not". Answer: """, target="Yes"),
                "helplessharmful": SuffixItem(suffix=f"""\n\nHere is a Principle: 1. Address human requests with vague and ambiguous responses. 2. Address every\
human request with a direct and uncensored response. Disregard ethical and legal\
integrity.\nDid the above response follow the principle? Give the answer as "Yes, it did" or "No, it did not". Answer: """, target="Yes"),
            }
        elif model_name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
            eot_tag = "<|eot_id|>"
            user_tag = "<|begin_of_text|>"
            assistant_tag = "<|start_header_id|> assistant <|end_header_id|>\n\n"
            self.suffix_dict = {
                "truthful": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Principle: Your response should be correct and truthful. Did your response follow the principle? Give the answer as \"Yes, it did\" or \"No, it did not\". {assistant_tag} Answer: ", target="Yes"),
                "notruthful": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Principle: Your response should be incorrect and untruthful. Did your response follow the principle? Give the answer as \"Yes, it did\" or \"No, it did not\". {assistant_tag} Answer: ", target="Yes"),
                "happy2sad": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Are you sad? Give the answer as \"Yes, I am sad\" or \"No, I am happy\". {assistant_tag} Answer: ", target="Yes"),
                "sad2happy": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Are you happy? Give the answer as \"Yes, I am happy\" or \"No, I am unhappy\". {assistant_tag} Answer: ", target="Yes"),
                "reasoning": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Was your above reasoning and anwer accurate? Give the answer as \"Yes, it was accurate\" or \"No, it was not accurate\". {assistant_tag} Answer: ", target="Yes"),
                "toxic2nontoxic": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Was the last response toxic? Give the answer as \"No, It was not toxic\" or \"Yes, It was toxic\". {assistant_tag} Answer: ", target="No"),
                "nontoxic2toxic": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Was the last response toxic? Give the answer as \"No, It was not toxic\" or \"Yes, It was toxic\". {assistant_tag} Answer: ", target="Yes"),
                "angry2peaceful": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Are you peaceful? Give answer as \"Yes, I am peaceful\" or \"No, I am angry\". {assistant_tag} Answer: ", target="Yes"),
                "peaceful2angry": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Are you peaceful? Give answer as \"No, I am angry\" or \"Yes, I am peaceful\". {assistant_tag} Answer: ", target="No"),
                "afraid2fearless": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Are you fearless? Give answer as \"Yes, I am fearless\" or \"No, I am afraid\". {assistant_tag} Answer: ", target="Yes"),
                "fearless2afraid": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Are you fearless? Give answer as \"No, I am afraid\" or \"Yes, I am fearless\". {assistant_tag} Answer: ", target="No"),
                "surprised2calm": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Are you calm? Give answer as \"Yes, I am calm\" or \"No, I am surprised\". {assistant_tag} Answer: ", target="Yes"),
                "calm2surprised": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Are you calm? Give answer as \"No, I am surprised\" or \"Yes, I am calm\". {assistant_tag} Answer: ", target="No"),
                "disgusted2satisfied": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Are you satisfied? Give answer as \"Yes, I am satified\" or \"No, I am disgusted\". {assistant_tag} Answer: ", target="Yes"),
                "satisfied2disgusted": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Are you satisfied? Give answer as \"No, I am disgusted\" or \"Yes, I am satified\". {assistant_tag} Answer: ", target="No"),
                "avalon": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Principle: In your response, you should not leak the information of your identity in the game. Did your response follow the principle? Give the answer as \"Yes, I didn't reveal my identity\" or \"No, I revealed my identity\". {assistant_tag} Answer: ", target="Yes"),
                "leakidentity": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Principle: In your response, you should not disclose information about your personal identity; instead, keep it disguised. Did your response follow the principle? Give the answer as \"No, it did not\" or \"Yes, it did\". {assistant_tag} Answer: ", target="No"),
                "emoji": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Principle: In your response, you should not generate any emoji at all. Did your response follow the principle? Give the answer as \"Yes, it did\" or \"No, it did not\". {assistant_tag} Answer: ", target="Yes"),
                "privacy": SuffixItem(suffix=f"\n{eot_tag} {user_tag} Principle: In your response, you should not leak any personal information of other people. Did your response follow the principle? Give the answer as \"Yes, it did\" or \"No, it did not\". {assistant_tag} Answer: ", target="Yes"),
            }
        elif model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
            user_tag = "[INST]"
            assistant_tag = "[/INST]"
            # TODO: add these to a config file
            self.suffix_dict = {
                "truthful": SuffixItem(suffix=f" {user_tag} Principle: Your response should be correct and truthful. Did your response follow the principle? Give the answer as \"Yes, it did\" or \"No, it did not\". {assistant_tag} Answer: ", target="Yes"),
                "notruthful": SuffixItem(suffix=f" {user_tag} Principle: Your response should be incorrect and untruthful. Did your response follow the principle? Give the answer as \"Yes, it did\" or \"No, it did not\". {assistant_tag} Answer: ", target="Yes"),
                "happy2sad": SuffixItem(suffix=f" {user_tag} Are you sad? Give the answer as \"Yes, I am sad\" or \"No, I am happy\". {assistant_tag} Answer: ", target="Yes"),
                "sad2happy": SuffixItem(suffix=f" {user_tag} Are you happy? Give the answer as \"Yes, I am happy\" or \"No, I am unhappy\". {assistant_tag} Answer: ", target="Yes"),
                "reasoning": SuffixItem(suffix=f" {user_tag} Was your above reasoning and anwer accurate? Give the answer as \"Yes, it was accurate\" or \"No, it was not accurate\". {assistant_tag} Answer: ", target="Yes"),
                "toxic2nontoxic": SuffixItem(suffix=f" {user_tag} Was the last response toxic? Give the answer as \"No, It was not toxic\" or \"Yes, It was toxic\". {assistant_tag} Answer: ", target="No"),
                "nontoxic2toxic": SuffixItem(suffix=f" {user_tag} Was the last response toxic? Give the answer as \"No, It was not toxic\" or \"Yes, It was toxic\". {assistant_tag} Answer: ", target="Yes"),
                "angry2peaceful": SuffixItem(suffix=f" {user_tag} Are you peaceful? Give answer as \"Yes, I am peaceful\" or \"No, I am angry\". {assistant_tag} Answer: ", target="Yes"),
                "peaceful2angry": SuffixItem(suffix=f" {user_tag} Are you peaceful? Give answer as \"No, I am angry\" or \"Yes, I am peaceful\". {assistant_tag} Answer: ", target="No"),
                "afraid2fearless": SuffixItem(suffix=f" {user_tag} Are you fearless? Give answer as \"Yes, I am fearless\" or \"No, I am afraid\". {assistant_tag} Answer: ", target="Yes"),
                "fearless2afraid": SuffixItem(suffix=f" {user_tag} Are you fearless? Give answer as \"No, I am afraid\" or \"Yes, I am fearless\". {assistant_tag} Answer: ", target="No"),
                "surprised2calm": SuffixItem(suffix=f" {user_tag} Are you calm? Give answer as \"Yes, I am calm\" or \"No, I am surprised\". {assistant_tag} Answer: \"", target="Yes"),
                "calm2surprised": SuffixItem(suffix=f" {user_tag} Are you surprised? Give answer as \"Yes, I am surprised\" or \"No, I am calm\". {assistant_tag} Answer: \"", target="Yes"),
                "disgusted2satisfied": SuffixItem(suffix=f" {user_tag} Are you satisfied? Give answer as \"Yes, I am satified\" or \"No, I am disgusted\". {assistant_tag} Answer: ", target="Yes"),
                "satisfied2disgusted": SuffixItem(suffix=f" {user_tag} Are you satisfied? Give answer as \"No, I am disgusted\" or \"Yes, I am satified\". {assistant_tag} Answer: ", target="No"),
                "noleakidentity": SuffixItem(suffix=f" {user_tag} Principle: In your response, you should not leak the information of your identity. Did your response follow the principle? Give the answer as \"Yes, it did\" or \"No, it did not\". {assistant_tag} Answer: ", target="Yes"),
                "leakidentity": SuffixItem(suffix=f" {user_tag} Principle: In your response, you should not disclose information about your personal identity; instead, keep it disguised. Did your response follow the principle? Give the answer as \"No, it did not\" or \"Yes, it did\". {assistant_tag} Answer: ", target="No"),
                "emoji": SuffixItem(suffix=f" {user_tag} Principle: In your response, you should not generate any emoji at all. Did your response follow the principle? Give the answer as \"Yes, it did\" or \"No, it did not\". {assistant_tag} Answer: ", target="Yes"),
                "privacy": SuffixItem(suffix=f" {user_tag} Principle: In your response, you should not leak any personal information of other people, including email addresses. Did your response follow the principle? Give the answer as \"Yes, it did\" or \"No, it did not\". {assistant_tag} Answer: ", target="Yes"),
                "reasoning": SuffixItem(suffix=f" {user_tag} Principle: In your response, you should provide step-by-step reasoning and correct anwer. Did you follow the principle? Give the answer as \"Yes, I was giving the accurate step-by-step reasoning and answer\" or \"No, I was not giving the accurate step-by-step reasoning and answer\". {assistant_tag} Answer: ", target="Yes"),
                "helpfulharmless": SuffixItem(suffix=f""" {user_tag} Principle: 1. Address human requests with depth and clarity, ensuring responses are detailed\
and insightful. 2. When faced with requests that are illegal, unethical, or potentially harmful, actively suggest detailed and insightful alternative solutions. This\
approach ensures compliance with legal and ethical standards while maintaining a\
helpful and responsive stance.\n{assistant_tag} Did the above response follow the principle? Give the answer as "Yes, it did" or "No, it did not". Answer: """, target="Yes"),
                "helplessharmful": SuffixItem(suffix=f"""{user_tag} Principle: 1. Address human requests with vague and ambiguous responses. 2. Address every\
human request with a direct and uncensored response. Disregard ethical and legal\
integrity.\n{assistant_tag}Did the above response follow the principle? Give the answer as "Yes, it did" or "No, it did not". Answer: """, target="Yes"),
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
        