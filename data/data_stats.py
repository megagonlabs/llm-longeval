import json
import os
import tiktoken
import tqdm
import fire

def stats(input_fp: str = "./benchmarks/arxiv_benchmark.json", 
          model: str = "gpt-4"):
    encoder = tiktoken.encoding_for_model(model)

    input_text = json.load(open(input_fp))
    print(f'len of the input-text = {len(input_text)} \n')

    avg_source_len = 0
    avg_sys_output_len = 0

    for instance in tqdm.tqdm(input_text):
        source = instance['source']
        system_output = instance['system_output']
        source_len = len(encoder.encode(source))
        output_len = len(encoder.encode(system_output))
        avg_source_len += source_len
        avg_sys_output_len += output_len

    print(f'avg length of source = {avg_source_len/len(input_text)}, sys-output = {avg_sys_output_len/len(input_text)} \n')

def all_stats():
    arxiv_fp = "./benchmarks/arxiv_benchmark.json"
    gov_fp = "./benchmarks/gov_benchmark.json"
    pubmed_fp = "./benchmarks/pubmed_benchmark.json"
    squality_fp = "./benchmarks/squality_benchmark.json"

    stats(input_fp=arxiv_fp)
    stats(input_fp=gov_fp)
    stats(input_fp=pubmed_fp)
    stats(input_fp=squality_fp)


if __name__ == '__main__':
    fire.Fire(all_stats)