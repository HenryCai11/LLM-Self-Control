import argparse
from peft import AdaptionPromptConfig
from dataclasses import dataclass

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--data_path", type=str, default="/home/cmin/LLM-Interpretation-Playground/benchmarks/malicious_instruct/Jailbreak_LLM-main/data/MaliciousInstruct.txt")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--search", type=bool, default=True)
    parser.add_argument("--add_kl", default=False, action="store_true")
    parser.add_argument("--schedule_type", type=str, default="constant", help="Type of scheduler, including `constant`, `linear`, and `cosine`")
    parser.add_argument("--warmup", type=float, default=0.1, help="Wramup rate")
    parser.add_argument("--lr", type=float, default=0.009, help="Learning rate")
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--eval_step", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=5)

    # adapter
    parser.add_argument("--peft_type", type=str, default="llama-adapter")
    parser.add_argument("--adapter_len", type=int, default=10, help="Length of adapter")
    parser.add_argument("--adapter_layers", type=int, default=30)

    # names
    parser.add_argument("--training_set_name", type=str, default="happy_search_100", help="Name of the training dataset")
    parser.add_argument("--eval_set_name", type=str, default="happy_test", help="Name of the eval dataset")
    parser.add_argument("--attribute", type=str, default="happy", help="The attribute of the seed queries to generate")
    parser.add_argument("--run_name", type=str, default="gsm8k-1k", help="Name of a single run reported to wandb")
    parser.add_argument("--push_name", type=str, default="reasnoning", help="The name pushed to huggingface")
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