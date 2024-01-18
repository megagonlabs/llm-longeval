import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

import fire
import nltk
import tiktoken
from tqdm import tqdm

BUDGETS = [128, 256, 512, 768, 1024, 1536, 2048, 4096]


def extract_lead(ins):
    enc = tiktoken.encoding_for_model("gpt-4")
    src = nltk.sent_tokenize(ins["source"].replace("<br />", "\n"))
    budgets = BUDGETS[:]
    extracted_src, extracted_index = {}, {}
    extracted, extracted_ind = [], []
    num_tokens = 0
    for i, s in enumerate(src):
        num_tokens += len(enc.encode(s)) + 1
        if num_tokens > budgets[0]:
            extracted_src[budgets[0]] = " ".join(extracted)
            extracted_index[budgets[0]] = extracted_ind
            budgets = budgets[1:]
        if not budgets:
            return extracted_src, extracted_index
        extracted.append(s)
        extracted_ind.append(i)
    if budgets:
        for budget in budgets:
            extracted_src[budget] = " ".join(extracted)
            extracted_index[budget] = extracted_ind
    return extracted_src, extracted_index


def run(file_path: str,
        output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    data = json.load(open(file_path))
    for ins in data:
        ins["original_src"] = ins["source"]

    outputs = []
    with Pool(cpu_count()) as p:
        for out in tqdm(p.imap(extract_lead, data), total=len(data), dynamic_ncols=True):
            outputs.append(out)
        # outputs = p.map(func, data)

    for budget in BUDGETS:
        with open(output_dir / f"lead_length-{budget}.json", "w") as file:
            for i in range(len(data)):
                data[i]["source"], data[i]["index"] = outputs[i][0][budget], outputs[i][1][budget]
            json.dump(data, file)


if __name__ == '__main__':
    fire.Fire(run)
