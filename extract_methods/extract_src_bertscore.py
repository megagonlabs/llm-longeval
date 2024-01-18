import bisect
import json
from datetime import datetime
from pathlib import Path

import fire
import nltk
import numpy as np
import tiktoken
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

BUDGETS = [128, 256, 512, 768, 1024, 1536, 2048, 4096]
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "microsoft/deberta-large-mnli"
TOKENIZER = AutoTokenizer.from_pretrained(model_name)
MODEL = AutoModel.from_pretrained(model_name).to(device).eval()
num_layers = 18
MODEL.encoder.layer = torch.nn.ModuleList([layer for layer in MODEL.encoder.layer[:num_layers]])


def extract_oracle(ins):
    enc = tiktoken.encoding_for_model("gpt-4")
    budgets = BUDGETS[:]
    total_src = len(enc.encode(ins["source"].replace("<br />", "\n")))
    src = nltk.sent_tokenize(ins["source"].replace("<br />", "\n"))
    src_index = list(range(len(src)))
    with torch.no_grad():
        print(datetime.now(), "Encode:")
        batch = TOKENIZER([ins["system_output"]], padding=True, return_tensors="pt").to(device)
        system_output_emb = MODEL(**batch).last_hidden_state[0].cpu()
        system_output_emb.div_(torch.norm(system_output_emb, dim=-1).unsqueeze(-1))
        system_output_masks = batch["attention_mask"].cpu()

        batch = TOKENIZER(src, padding=True, return_tensors="pt").to(device)
        src_sent_emb = MODEL(**batch).last_hidden_state.cpu()
        src_sent_emb.div_(torch.norm(src_sent_emb, dim=-1).unsqueeze(-1))
        src_sent_masks = batch["attention_mask"].cpu()
        print(datetime.now(), "Sim")
        sim = src_sent_emb @ system_output_emb.T
        masks = src_sent_masks.unsqueeze(2).float() @ system_output_masks.unsqueeze(0).float()
        sim = (sim * masks).cpu().numpy()
        print(datetime.now(), "Sim finish")

    num_tokens = 0
    extracted_src, extracted_index = {}, {}
    cand = []
    prev_score = 0.
    do_pass = False
    skip = set()
    while True:
        used = set(cand)
        src_index_unused = [i for i in src_index if i not in used]
        if not src_index_unused:
            for budget in budgets:
                extracted_src[budget] = " ".join(src[i] for i in cand)
                extracted_index[budget] = cand
            break
        if do_pass:
            i = min(src_index_unused)
        else:
            c = []
            for i in src_index_unused:
                if i not in skip:
                    x = cand[:]
                    bisect.insort(x, i)
                    c.append(x)
            cand_sim = np.stack([np.concatenate([sim[i] for i in ix]) for ix in c])
            # import pdb; pdb.set_trace()
            word_recall = cand_sim.max(axis=1)
            scores = word_recall.mean(axis=1)
            if scores.max() == prev_score:
                do_pass = True
            skip.update([i for i, t in enumerate(scores == prev_score) if t])

            prev_score = scores.max()
            print("Recall: ", scores.max())
            i = src_index_unused[scores.argmax()]
        print("Num_tokens: ", num_tokens, " / ", total_src)

        num_tokens += len(enc.encode(src[i] + "\n"))
        if num_tokens > budgets[0]:
            extracted_src[budgets[0]] = " ".join(src[i] for i in cand)
            extracted_index[budgets[0]] = cand
            budgets = budgets[1:]
            if not budgets:
                break
        bisect.insort(cand, i)
    print(datetime.now(), "End")
    return extracted_src, extracted_index


def run(file_path: str,
        output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    data = json.load(open(file_path))
    for ins in data:
        ins["original_src"] = ins["source"]

    outputs = []
    # with Pool(cpu_count()) as p:
    #     for out in tqdm(p.imap(extract_oracle, data), total=len(data), dynamic_ncols=True):
    #         outputs.append(out)
    #     # outputs = p.map(func, data)
    for ins in tqdm(data, dynamic_ncols=True):
        outputs.append(extract_oracle(ins))

    for budget in BUDGETS:
        with open(output_dir / f"bertscore_length-{budget}.json", "w") as file:
            for i in range(len(data)):
                data[i]["source"], data[i]["index"] = outputs[i][0][budget], outputs[i][1][budget]
            json.dump(data, file)


if __name__ == '__main__':
    fire.Fire(run)
