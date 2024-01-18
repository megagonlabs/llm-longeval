import json
import os
import re

import openai
import tiktoken
import tqdm
from tenacity import retry, wait_random_exponential


def parse_score(output):
    matched = re.search(r"^ ?(\d+)", output)
    if matched:
        score = float(matched.group(1))
    else:
        score = 0.
    return score


class Evaluator:
    def __init__(self,
                 model: str,
                 api_key: str = None,
                 organization: str = None):
        self.enc = tiktoken.encoding_for_model(model)
        self.model = model
        assert model.startswith("gpt-3.5") or model.startswith("gpt-4")
        openai.organization = organization
        openai.api_key = os.getenv("OPENAI_API_KEY") if api_key is None else api_key

    @retry(wait=wait_random_exponential(min=1, max=40))
    def score(self,
              content: str,
              temperature: float = 0.,
              n: int = 1):
        responses = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": content}],
            temperature=temperature,
            max_tokens=8,
            n=n)
        scores = [parse_score(x["message"]["content"]) for x in responses["choices"]]
        return sum(scores) / len(scores)

    def eval_output(self,
                    input_text_fp: str,
                    prompt_fp: str,
                    temperature: float = 0.,
                    n: int = 1):
        input_text = json.load(open(input_text_fp))
        print(f'number of docs = {len(input_text)}')
        prompt = open(prompt_fp).read()
        prompt_len = len(self.enc.encode(prompt))

        eval_results = []
        for instance in tqdm.tqdm(input_text):
            source = instance['source']
            system_output = instance['system_output']

            source_len = len(self.enc.encode(source))
            output_len = len(self.enc.encode(system_output))

            if self.model.startswith("gpt-4"):
                # gpt-4-0613 only allows 8k input token length
                if prompt_len + source_len + output_len + 1 > 8000:
                    source_len = 8000 - prompt_len - output_len - 1
                    source = self.enc.decode(self.enc.encode(source[:source_len]))
            elif self.model.startswith("gpt-3.5-turbo-16k"):
                #  gpt-3.5-turbo-16k-0613 only allows 16k input token length
                if prompt_len + source_len + output_len + 1 > 16000:
                    source_len = 16000 - prompt_len - output_len - 1
                    source = self.enc.decode(self.enc.encode(source[:source_len]))
            else:
                raise ValueError()

            current_prompt = prompt.replace('{{article}}', source).replace('{{summary}}', system_output)
            instance["prompt"] = current_prompt
            instance["model_score"] = self.score(current_prompt, temperature=temperature, n=n)
            eval_results.append(instance)

        return eval_results
