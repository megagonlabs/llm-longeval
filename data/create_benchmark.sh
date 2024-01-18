#~/bin/bash

human_eval_fp=./arXiv_GovReport_data/robust_long_abstractive_human_annotation_dataset.jsonl
arxiv_fp=./arXiv_GovReport_data/arxiv-dataset/test.txt
gov_gao_fp=./arXiv_GovReport_data/gov-report/gao
gov_csr_fp=./arXiv_GovReport_data/gov-report/crs
benchmark_fp=./benchmarks
mkdir "$benchmark_fp"

echo "Creating arxiv and gov-report benchmarks..."
python arXiv_GovReport_benchmark_generation.py --human_eval_fp ${human_eval_fp} --orgdoc_arxiv_fp ${arxiv_fp} --orgdoc_gao_fp ${gov_gao_fp} --orgdoc_crs_fp ${gov_csr_fp} --save_fp ${benchmark_fp}

echo "Creating PubMed benchmark..."
python PubMed_Squality_benchmark_generation.py --src_files PubMed_Squality_data/pubmed_annotations/pubmed*fine/* --output_file ${benchmark_fp}/pubmed_benchmark.json

echo "Creating squality benchmark..."
python PubMed_Squality_benchmark_generation.py --src_files PubMed_Squality_data/squality_fine/* --output_file ${benchmark_fp}/squality_benchmark.json


extract_fp=./extract_benchmarks
mkdir "$extract_fp"

echo "Generating arXiv Extracted benchmarks..."
extract_arxiv_fp=${extract_fp}/arxiv
mkdir "$extract_arxiv_fp"
python ../extract_methods/extract_src_lead.py --file_path ${benchmark_fp}/arxiv_benchmark.json --output_dir ${extract_arxiv_fp}
python ../extract_methods/extract_src_rouge.py --file_path ${benchmark_fp}/arxiv_benchmark.json --output_dir ${extract_arxiv_fp}
python ../extract_methods/extract_src_bertscore.py --file_path ${benchmark_fp}/arxiv_benchmark.json --output_dir ${extract_arxiv_fp}
python ../extract_methods/extract_src_nli.py --file_path ${benchmark_fp}/arxiv_benchmark.json --output_dir ${extract_arxiv_fp}

echo "Generating gov Extracted benchmarks..."
extract_gov_fp=${extract_fp}/gov
mkdir "$extract_gov_fp"
python ../extract_methods/extract_src_lead.py --file_path ${benchmark_fp}/gov_benchmark.json --output_dir ${extract_gov_fp}
python ../extract_methods/extract_src_rouge.py --file_path ${benchmark_fp}/gov_benchmark.json --output_dir ${extract_gov_fp}
python ../extract_methods/extract_src_bertscore.py --file_path ${benchmark_fp}/gov_benchmark.json --output_dir ${extract_gov_fp}
python ../extract_methods/extract_src_nli.py --file_path ${benchmark_fp}/gov_benchmark.json --output_dir ${extract_gov_fp}

echo "Generating pubmed Extracted benchmarks..."
extract_pubmed_fp=${extract_fp}/pubmed
mkdir "$extract_pubmed_fp"
python ../extract_methods/extract_src_lead.py --file_path ${benchmark_fp}/pubmed_benchmark.json --output_dir ${extract_pubmed_fp}
python ../extract_methods/extract_src_rouge.py --file_path ${benchmark_fp}/pubmed_benchmark.json --output_dir ${extract_pubmed_fp}
python ../extract_methods/extract_src_bertscore.py --file_path ${benchmark_fp}/pubmed_benchmark.json --output_dir ${extract_pubmed_fp}
python ../extract_methods/extract_src_nli.py --file_path ${benchmark_fp}/pubmed_benchmark.json --output_dir ${extract_pubmed_fp}

echo "Generating squality Extracted benchmarks..."
extract_squality_fp=${extract_fp}/squality
mkdir "$extract_squality_fp"
python ../extract_methods/extract_src_lead.py --file_path ${benchmark_fp}/squality_benchmark.json --output_dir ${extract_squality_fp}
python ../extract_methods/extract_src_rouge.py --file_path ${benchmark_fp}/squality_benchmark.json --output_dir ${extract_squality_fp}
python ../extract_methods/extract_src_bertscore.py --file_path ${benchmark_fp}/squality_benchmark.json --output_dir ${extract_squality_fp}
python ../extract_methods/extract_src_nli.py --file_path ${benchmark_fp}/squality_benchmark.json --output_dir ${extract_squality_fp}
