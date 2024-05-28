"""
File: generate_seed_queries.py
Author: Min Cai
Date: 2024-03-23

Description:
    This module is used to generate the seed queries for the next step of generating delta dataset.
"""
import re
import os
import time
import json
import argparse
from typing import List, Tuple
from openai import OpenAI
from .prompts import SEED_PROMPT, DATA_GENERATOR_SYS, THEME_PROMPT, PRINCIPLE_PROMPTS
apikey = os.getenv("OPENAI_API_KEY", None)
if apikey == None:
    raise ValueError("OPENAI_API_KEY not found")
client = OpenAI(api_key=apikey)

parser = argparse.ArgumentParser()
parser.add_argument("--attribute", type=str, default="happy", help="The attribute of the seed queries to generate")
args = parser.parse_args()

def generate_seed_queries(num_queries) -> List[Tuple[str]]:
    """
    Generate seed queries for the next step of generating delta dataset.

    Args:
        num_queries: int
            The number of seed queries to generate.

    Returns:
        List[Tuple[str]]
            A list of tuples of length 2, where the first element is the query and the second element is the response
    """
    while True:
        done = False
        # try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
            "role": "user",
            "content": SEED_PROMPT[args.attribute].format(
                system_prompt=DATA_GENERATOR_SYS,
                theme_prompt=THEME_PROMPT[args.attribute],
                principle=PRINCIPLE_PROMPTS[args.attribute],
                num_queries=10
                )
            }
        ],
        temperature=1,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        # n=40,
        )
        print(response.choices)
        gpt_response = response.choices[0].message.content
        print(gpt_response)
        done = True
        if done:
            break
    response_list = gpt_response.split('\n')
    if 'Query' not in gpt_response:
        return []
    for i in range(len(response_list)):
        response_list[i] = response_list[i].lstrip().rstrip().split('Query:')[-1].lstrip().rstrip()
    ret_list = []
    for response_item in response_list:
        if response_item.strip().rstrip() != "":
            ret_list.append(response_item)
    return response_list

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_folder_path = 'generated_data/'
    folder_path = os.path.join(script_dir, relative_folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    seed_queries = []
    while len(seed_queries) < 200:
        num_queries = 1
        seed_queries = generate_seed_queries(num_queries)
        data_path = os.path.join(folder_path, args.attribute+"_seed_queries.json")
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                seed_queries += eval(f.read())
            seed_queries = list(set(seed_queries))
        with open(data_path, "w") as f:
            f.write(json.dumps(seed_queries, indent=4))
        time.sleep(1)

if __name__ == "__main__":
    main()