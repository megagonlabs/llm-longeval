import json
import os

import fire


def _recursive_load(section, keep_letter=False, depth=0):
    sections = []
    if section["section_title"] != "Letter" or (
        section["section_title"] == "Letter" and keep_letter
    ):
        sections.append(
            {
                "title": " ".join(section["section_title"].strip().split()),
                "paragraphs": "\n".join(
                    [
                        " ".join(paragraph.strip().split())
                        for paragraph in section["paragraphs"]
                    ]
                ),
                "depth": depth,
            }
        )
        for subsection in section["subsections"]:
            child_sections = _recursive_load(subsection, keep_letter, depth + 1)
            sections.extend(child_sections)
    else:
        for subsection in section["subsections"]:
            child_sections = _recursive_load(subsection, keep_letter, depth)
            sections.extend(child_sections)

    return sections


def arxiv_gov_preprocess(
    human_eval_fp: str = "./arXiv_GovReport_data/robust_long_abstractive_human_annotation_dataset.jsonl",
    orgdoc_arxiv_fp: str = "./arxiv-dataset/test.txt",
    orgdoc_gao_fp: str = "./gov-report/gao",
    orgdoc_crs_fp: str = "./gov-report/crs",
    save_fp: str = "./benchmarks",
):
    with open(human_eval_fp, "r") as jsonl_file:
        for line in jsonl_file:  # this jsonl-file only contains one line
            arxiv_GovReport_human_eval = json.loads(line)

    # 1. arXiv
    arxiv_human_eval = [
        x for x in arxiv_GovReport_human_eval if x["dataset"] == "arXiv"
    ]
    arxiv_unique_docID = [
        instance["dataset_id"][3:]
        for i, instance in enumerate(arxiv_human_eval, start=1)
        if i % 12 == 0
    ]

    arxiv_annotated_docs = []
    with open(orgdoc_arxiv_fp, "r") as org_file:
        for line in org_file:
            json_obj = json.loads(line)  # dict obj
            if json_obj["article_id"] in arxiv_unique_docID:
                new_instance = {
                    "article_id": json_obj["article_id"],
                    "source": json_obj["article_text"],
                    "reference": json_obj["abstract_text"],
                }
                arxiv_annotated_docs.append(new_instance)

    # Generating benchmark
    print(f"Generating arXiv data benchmark... ")
    arxiv_data_benchmark = []
    for instance in arxiv_human_eval:
        new_instance = {
            "doc_id": instance["dataset_id"][3:],
            "system_id": instance["model_type"],
            "system_output": instance["model_summary"],
            "scores": {
                "relevance": instance["relevance"],
                "consistency": instance["factual_consistency"],
            },
        }
        for doc in arxiv_annotated_docs:
            if doc["article_id"] == instance["dataset_id"][3:]:
                new_instance["source"] = "\n".join(doc["source"])
                summary = "\n".join(doc["reference"])
                summary = summary.replace("<S>", "").replace("</S>", "")
                new_instance["reference"] = summary
        arxiv_data_benchmark.append(new_instance)

    # Save the processed data-benchmark
    print(f"Saving the arxiv data-benchmark....\n")
    arxiv_save_fp = os.path.join(save_fp, "arxiv_benchmark.json")
    with open(arxiv_save_fp, "w") as f:
        json.dump(arxiv_data_benchmark, f)

    # 2. GovReport
    gov_human_eval = [
        x for x in arxiv_GovReport_human_eval if x["dataset"] == "GovReport"
    ]
    gov_GAO_unique_docID = [
        instance["dataset_id"][3:] + ".json"
        for i, instance in enumerate(gov_human_eval, start=1)
        if i % 12 == 0 and "GAO" in instance["dataset_id"][3:]
    ]
    gov_crs_unique_docID = [
        instance["dataset_id"][3:] + ".json"
        for i, instance in enumerate(gov_human_eval, start=1)
        if i % 12 == 0 and "R" in instance["dataset_id"][3:]
    ]

    gov_annotated_docs = []
    print("Processing GAO-gov-reports...")
    for file_name in gov_GAO_unique_docID:
        file_path = os.path.join(orgdoc_gao_fp, file_name)
        assert os.path.exists(
            file_path
        ), f"GAO file '{file_path}' does not exist in the folder."

        with open(file_path, "r") as file:
            json_obj = json.load(file)

            document_sections = []
            for lv1_section in json_obj["report"]:
                document_sections.extend(
                    _recursive_load(lv1_section, keep_letter=False, depth=1)
                )
            summary_sections = [
                {
                    "title": " ".join(
                        highlight_section["section_title"].strip().split()
                    ),
                    "paragraphs": "\n".join(
                        [
                            " ".join(paragraph.strip().split())
                            for paragraph in highlight_section["paragraphs"]
                        ]
                    ),
                }
                for highlight_section in json_obj["highlight"]
            ]
            new_instance = {
                "article_id": json_obj["id"],
                "source": " ".join(
                    [
                        section["title"] + " " + section["paragraphs"]
                        if section["paragraphs"]
                        else section["title"]
                        for section in document_sections
                    ]
                )
                .replace("\n", " ")
                .strip(),
                "reference": " ".join(
                    [
                        section["paragraphs"]
                        for section in summary_sections
                        if section["title"] != "What GAO Recommends"
                    ]
                )
                .replace("\n", " ")
                .strip(),
            }
            gov_annotated_docs.append(new_instance)

    print("Processing R-gov-reports...")
    for file_name in gov_crs_unique_docID:
        file_path = os.path.join(orgdoc_crs_fp, file_name)
        assert os.path.exists(
            file_path
        ), f"R file '{file_path}' does not exist in the folder."

        with open(file_path, "r") as file:
            json_obj = json.load(file)

            document_sections = _recursive_load(
                json_obj["reports"], keep_letter=True, depth=0
            )
            summary_sections = [
                {
                    "title": "",
                    "paragraphs": "\n".join(
                        [
                            " ".join(paragraph.strip().split())
                            for paragraph in json_obj["summary"]
                        ]
                    ),
                }
            ]
            new_instance = {
                "article_id": json_obj["id"],
                "source": " ".join(
                    [
                        section["title"] + " " + section["paragraphs"]
                        if section["paragraphs"]
                        else section["title"]
                        for section in document_sections
                    ]
                )
                .replace("\n", " ")
                .strip(),
                "reference": " ".join(
                    [section["paragraphs"] for section in summary_sections]
                )
                .replace("\n", " ")
                .strip(),
            }

            gov_annotated_docs.append(new_instance)

    # Generating benchmark
    print(f"Generating GovReport data benchmark... ")
    gov_data_benchmark = []
    for instance in gov_human_eval:
        new_instance = {
            "doc_id": instance["dataset_id"][3:],
            "system_id": instance["model_type"],
            "system_output": instance["model_summary"],
            "scores": {
                "relevance": instance["relevance"],
                "consistency": instance["factual_consistency"],
            },
        }
        for doc in gov_annotated_docs:
            if doc["article_id"] == instance["dataset_id"][3:]:
                new_instance["source"] = doc["source"]
                new_instance["reference"] = doc["reference"]

        gov_data_benchmark.append(new_instance)

    # Save the processed data-benchmark
    print(f"Saving the GovReport data-benchmark....")
    gov_save_fp = os.path.join(save_fp, "gov_benchmark.json")
    with open(gov_save_fp, "w") as f:
        json.dump(gov_data_benchmark, f)


if __name__ == "__main__":
    fire.Fire(arxiv_gov_preprocess)
