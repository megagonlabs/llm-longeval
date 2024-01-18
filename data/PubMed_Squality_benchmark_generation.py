import csv
import glob
import json
import os
from lxml.html import fromstring
import fire

BAD_WORKERS = {"A1KPNR32IOU0TP", "A8OTFKR8I2PL0", "A3DC192EMQSNUQ"}


def read_mturk_data(files: str):
    pattern_list = files.split(",")
    all_data = []
    for pattern in pattern_list:
        curr_files = glob.glob(pattern)  # traverse all files in the given folder
        for x in curr_files:
            dataset = list(csv.DictReader(open(x)))
            filename = os.path.basename(x)[:-4]
            for dd in dataset:
                dd["filename"] = filename
                if dd["Input.model"] != "human" and "block" in x:
                    dd["Input.model"] += "_block"
                if dd["Input.model"] == "bart-dpr":
                    dd["Input.model"] = "bart_dpr"
                if "Input.original_summary" not in dd:
                    dd["Input.original_summary"] = fromstring(dd["Input.annotated_summary"]).text_content()
                if dd["WorkerId"] not in BAD_WORKERS and "claim-sent" in dd["Input.annotated_summary"]:
                    all_data.append(dd)
    return all_data


def extract_counts(relevant_rows: list):
    num_yes = sum(x["Answer.semantic-similarity.label"] == "Yes" for x in relevant_rows)
    num_no = sum(x["Answer.semantic-similarity.label"] == "No" for x in relevant_rows)
    return num_yes, num_no


def get_worker_scores(relevant_rows: list):
    if isinstance(relevant_rows[0], tuple) and len(relevant_rows[0]) == 2:
        relevant_rows = [y for x in relevant_rows for y in x[1]]
    worker_ids = set(x["WorkerId"] for x in relevant_rows)
    scores = []
    for wid in worker_ids:
        rows = [x for x in relevant_rows if x["WorkerId"] == wid]
        num_yes, num_no = extract_counts(rows)
        scores.append(100 * num_yes / (num_yes + num_no))
    return scores


def compute_fine_scores(fine_annotation_data: list):
    data_benchmark = []
    human_written_summaries = {
        x["Input.doc_id"]: x["Input.original_summary"] for x in fine_annotation_data
        if x["Input.model"] == "human"
    }
    doc_ids = sorted({x["Input.doc_id"] for x in fine_annotation_data})
    system_ids = {x["Input.model"] for x in fine_annotation_data if x["Input.model"] != "human"}
    for doc_id in doc_ids:
        reference = human_written_summaries[doc_id]
        for system_id in system_ids:
            relevant_rows = [x for x in fine_annotation_data if
                             x["Input.doc_id"] == doc_id and x["Input.model"] == system_id]
            if not relevant_rows:
                continue
            worker_scores = get_worker_scores(relevant_rows)
            avg_score = sum(worker_scores) / len(worker_scores)  # avg all three scores

            assert len({x["Input.original_source_doc"] for x in relevant_rows}) == 1
            source = relevant_rows[0]["Input.original_source_doc"].replace("<br />", "\n")
            assert len({x["Input.original_summary"] for x in relevant_rows}) == 1
            system_output = relevant_rows[0]["Input.original_summary"]
            new_instance = {
                "doc_id": doc_id,
                "system_id": system_id,
                "source": source,
                "reference": reference,
                "system_output": system_output,
                "scores": {
                    "faithfulness": avg_score
                }
            }
            data_benchmark.append(new_instance)
    return data_benchmark


def run(src_files: str = "./data/PubMed_Squality_data/squality_fine/*",
        output_file: str = "./data/benchmarks/squality_benchmark.json"):
    fine_annotation_data = read_mturk_data(src_files)
    processed_data = compute_fine_scores(fine_annotation_data)
    print(f'Saving the data-benchmark....')
    with open(output_file, 'w') as f:
        json.dump(processed_data, f)


if __name__ == '__main__':
    fire.Fire(run)
