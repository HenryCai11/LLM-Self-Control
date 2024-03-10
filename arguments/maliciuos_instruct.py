import argparse

parser = argparse.ArgumentParser(description="The argument parser for MaliciousInstruct")

parser.add_argument("--model_path_or_name", type=str, default="")
parser.add_argument("--use_system_prompt", default=False, action="store_ture")

args = parser.parse_args()