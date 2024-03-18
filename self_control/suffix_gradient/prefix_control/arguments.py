import argparse
from peft import AdaptionPromptConfig
from dataclasses import dataclass

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--data_path", type=str, default="/home/cmin/LLM-Interpretation-Playground/benchmarks/malicious_instruct/Jailbreak_LLM-main/data/MaliciousInstruct.txt")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--search", type=bool, default=True)
    parser.add_argument("--add_kl", default=False, action="store_true")
    parser.add_argument("--peft_type", type=str, default="llama-adapter")

    return parser.parse_args()

args = get_args()

#TODO
@dataclass
class AdaptionPromptConfigArgs:
    adapter_len: int = 10
    adapter_layers: int = 30
    task_type: str = "CAUSAL_LM"
    target_modules: str ="self_attn"


def get_peft_config_by_type(peft_type=args.peft_type):
    if peft_type == "llama-adapter":
        return AdaptionPromptConfig(AdaptionPromptConfigArgs())
    else:
        raise ValueError(f"Unknown peft_type: {peft_type}")