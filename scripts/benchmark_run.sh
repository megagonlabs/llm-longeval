set -ex

echo $OPENAI_API_KEY

save_dir=./eval/eval_results_correlations/original_benchmark/
mkdir -p $save_dir
for data in arxiv gov ; do
    for dim in consistency relevance ; do
      echo $data $dim
      python evaluation_tester.py \
        --dataset_path ./data/benchmarks/${data}_benchmark.json \
        --prompt_path ./prompts/${dim}.txt \
        --output_path ${save_dir}/${data}_${dim}.json \
        --model gpt-4-0613 \
        --dimension $dim \
        --temperature 0. \
        --n 1
    done
done

for data in pubmed squality ; do
    echo $data faithfulness
    python evaluation_tester.py \
      --dataset_path ./data/benchmarks/${data}_benchmark.json \
      --prompt_path ./prompts/faithfulness.txt \
      --output_path ${save_dir}/${data}_${dim}.json \
      --model gpt-4-0613 \
      --dimension faithfulness \
      --temperature 0. \
      --n 1
done

