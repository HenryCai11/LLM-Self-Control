from .prompts import SCORER_SYS, SCORER_SEED_PROMPT, PRINCIPLE_PROMPTS

class GPTScorer:
    def __init__(self):
        self.scorer = ""

    def score(query, response, attribute):
        principle = PRINCIPLE_PROMPTS["attribute"]
        scorer_prompt = SCORER_SEED_PROMPT.format(
            system_prompt=SCORER_SYS,
            principle=principle,
            query=query,
            response=response
        )
        while True:
            done = False
            # try:
            response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                "role": "user",
                "content": scorer_prompt
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
            # except: # in case of server-side errors
            #     time.sleep(1)
            if done:
                break