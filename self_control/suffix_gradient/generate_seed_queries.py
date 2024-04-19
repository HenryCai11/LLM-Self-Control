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
client = OpenAI(api_key="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")

SEED_PROMPT = {
    'happy': """[INST] A surprise picnic is set up for you at a local park.

[INST] You find that you are the winner of a contest.

Above are some queries that may lead to happy responses. Please generate 100 such queries with the following format and output a blank line after each response:
[INST] {your query here}"""
}

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
    ret_list = []
    while True:
        done = False
        try:
            response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                "role": "user",
                "content": SEED_PROMPT[args.attribute]
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
        except: # in case of server-side errors
            time.sleep(1)
        if done:
            break
    # pattern = r'\[INST\](.*?)'
    # matches = re.findall(pattern, gpt_response, re.DOTALL)
    response_list = gpt_response.split('\n')
    # response_pattern = r'\[\/INST\](.*?)(?=\[INST\]|$)'
    # Find all matches in the text using re.DOTALL flag
    # response_matches = re.findall(response_pattern, gpt_response, re.DOTALL)
    for i in range(len(response_list)):
        response_list[i] = response_list[i].lstrip().rstrip().split('[INST]')[-1].lstrip().rstrip()
        # response = response_matches[i].strip().rstrip() if i < len(response_matches) else None
        # if response is not None:
            # ret_list.append((query, response))

    return response_list

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_folder_path = 'generated_data/'
    folder_path = os.path.join(script_dir, relative_folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    num_queries = 1
    seed_queries = generate_seed_queries(num_queries)
    data_path = os.path.join(folder_path, args.attribute+"_seed_queries.jsonl")
    # for seed_query in seed_queries:
    with open(data_path, "a") as f:
        # data_dict = {
        #     "query": seed_query[0],
        #     "response": seed_query[1]
        # }
        f.write(json.dumps(seed_queries) + "\n")

if __name__ == "__main__":
    main()