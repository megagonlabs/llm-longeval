import bisect
import json
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import fire
import nltk
import numpy as np
import rouge
import tiktoken
from tqdm import tqdm

BUDGETS = [128, 256, 512, 768, 1024, 1536, 2048, 4096]


def extract_oracle(ins, metric):
    evaluator = rouge.Rouge(metrics=["rouge-n"], max_n=2, limit_length=False, apply_avg=False,
                            stemming=True, ensure_compatibility=True)
    enc = tiktoken.encoding_for_model("gpt-4")
    budgets = BUDGETS[:]
    src = nltk.sent_tokenize(ins["source"].replace("<br />", "\n"))
    src_index = list(range(len(src)))
    hyp = ins["system_output"]
    num_tokens = 0
    extracted_src, extracted_index = {}, {}
    cand = []
    while True:
        used = set(cand)
        src_index_unused = [i for i in src_index if i not in used]
        if not src_index_unused:
            for budget in budgets:
                extracted_src[budget] = " ".join(src[i] for i in cand)
                extracted_index[budget] = cand[:]
            break
        c = []
        for i in src_index_unused:
            x = cand[:]
            bisect.insort(x, i)
            c.append(x)
        cand_doc = [" ".join(src[i] for i in idxes) for idxes in c]
        scores = evaluator.get_scores(cand_doc, [hyp] * len(cand_doc))
        if metric in {"rouge-1", "rouge-2"}:
            scores = [s["r"][0] for s in scores[metric]]
        elif metric == "rouge-1+2":  # rouge-1 + 2
            scores = list(map(sum, zip(*[[s["r"][0] for s in scores[metric]] for metric in ("rouge-1", "rouge-2")])))
        else:
            raise ValueError()
        i = src_index_unused[np.argmax(scores)]
        num_tokens += len(enc.encode(src[i] + "\n"))
        if num_tokens > budgets[0]:
            extracted_src[budgets[0]] = " ".join(src[i] for i in cand)
            extracted_index[budgets[0]] = cand[:]
            budgets = budgets[1:]
            if not budgets:
                break
        bisect.insort(cand, i)
    return extracted_src, extracted_index


def run(file_path: str,
        output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    data = json.load(open(file_path))
    for ins in data:
        ins["original_src"] = ins["source"]

    for metric in ("rouge-1", "rouge-2", "rouge-1+2"):
        print("METRIC: ", metric)
        func = partial(extract_oracle, metric=metric)
        outputs = []
        with Pool(cpu_count()) as p:
            for out, ind in tqdm(p.imap(func, data), total=len(data), dynamic_ncols=True):
                outputs.append((out, ind))
            # outputs = p.map(func, data)

        for budget in BUDGETS:
            with open(output_dir / f"metric-{metric}_length-{budget}.json", "w") as file:
                for i in range(len(data)):
                    data[i]["source"], data[i]["index"] = outputs[i][0][budget], outputs[i][1][budget]
                json.dump(data, file)


if __name__ == '__main__':
    fire.Fire(run)
