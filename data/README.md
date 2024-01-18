# Long Document Summarization Benchmark 
Here we collect four publicly avaliable long summarization benchmarks along with their human-evaluation annotations. 

The following table shows the data benchmarks along with their scores types. 


## Source Human-evaluation Annotated Data
Get the PubMed and SQuALITY data from LongEval official repo: [LongEval: Guidelines for Human Evaluation of Faithfulness in Long-form Summarization](https://github.com/martiansideofthemoon/longeval-summarization/tree/master)
 - Annotated PubMed and SQuALITY: [Download](https://drive.google.com/drive/folders/1nLVmPQMmX_XOHrc_0I7oJBJfl6EMRqeK), put downloaded files in ```./PubMed_Squality_data```
 - Here we only take the fine human annotations, the coarse human annotations are not used. 

Get the arXiv and GovReport original source data and human annotated data from [How Far are We from Robust Long Abstractive Summarization?](https://arxiv.org/pdf/2210.16732.pdf) paper's [official repo](https://github.com/huankoh/How-Far-are-We-from-Robust-Long-Abstractive-Summarization):
 - ArXiv dataset: [Download](https://drive.google.com/file/d/1b3rmCSIoh6VhD4HKWjI4HOW-cSwcwbeC/view)
 - GovReport dataset: [Download](https://drive.google.com/file/d/1ik8uUVeIU-ky63vlnvxtfN2ZN-TUeov2/view)
 - Human annotated data ```robust_long_abstractive_human_annotation_dataset.jsonl``` [Download](https://github.com/huankoh/How-Far-are-We-from-Robust-Long-Abstractive-Summarization/blob/main/robust_long_abstractive_human_annotation_dataset.jsonl)
 - Please remember to put arxiv, GovReport and ```robust_long_abstractive_human_annotation_dataset.jsonl``` into ```./arXiv_GovReport_data```

## Benchmark Generation
Run the script:
```
chmod +x create_benchmark.sh
./create_benchmark.sh
```
<!-- Generate arXiv and GovReports benchmarks from original source documents (get original source data from [ArXiv](https://drive.google.com/file/d/1b3rmCSIoh6VhD4HKWjI4HOW-cSwcwbeC/view) and [GovReport](https://drive.google.com/file/d/1ik8uUVeIU-ky63vlnvxtfN2ZN-TUeov2/view)): 
```
python ./arXiv_GovReport_benchmark_generation.py --human_eval_fp ./arXiv_GovReport_data/robust_long_abstractive_human_annotation_dataset.jsonl --orgdoc_arxiv_fp ./arxiv-dataset/test.txt" --orgdoc_gao_fp ./gov-report/gao --orgdoc_crs_fp ./gov-report/crs --save_fp ./benchmarks
```

Generate PubMed benchmark:
```
python ./PubMed_Squality_benchmark_generation.py --src_files ./PubMed_Squality_data/pubmed_annotations/pubmed*fine/* --output_file ./benchmarks/pubmed_benchmark.json" 
```

Generate SQuALITY benchmark:
```
python ./PubMed_Squality_benchmark_generation.py --src_files ./PubMed_Squality_data/squality_fine/* --output_file ./benchmarks/squality_benchmark.json
``` -->

## Truncated source of the benchmarks
Four extracting methods:
 - Lead-X: select the first N tokens from the source document are selected (consisting of complete sentences). This is considered a strong baseline for extractive summarization
 - ROUGE-1/2/1+2 coverage: select sentences from the source document that maximize recall of ROUGE score with respect to the summary, until it reaches N tokens
 - BERTScore: select sentences as in ROUGE, but use the recall of BERTScore with respect to the summary, until it reaches N tokens
 - NLI: select sentences that are entailing or contradcting to the summary,  until it reaches N tokens


Find the dataset here [extracted_source](./extract_benchmarks)

## Benchmark Dataset Statistics
Run the script to get benchmark stats
```
python data_stats.py
```
