set -ex
# Run G-eval on all source-text-truncated long-document benchmarks
# Run all truncated methods
echo "RUNNING ALL TRUNCATED METHODS..."
set -ex

echo $OPENAI_API_KEY

save_dir=./eval/eval_results_correlations/gpt4/truncated_benchmark/
mkdir -p $save_dir
for data in arxiv gov ; do
    for dim in consistency relevance ; do
      for file in ./data/truncated_benchmarks/${data}/*.json ; do
        echo $data $dim $file
        python evaluation_tester.py \
          --dataset_path $file \
          --prompt_path ./prompts/${dim}.txt \
          --output_path ${save_dir}/${data}/${dim}/$(basename ${file}) \
          --model gpt-4-0613 \
          --dimension $dim \
          --temperature 0. \
          --n 1
      done
    done
done

for data in pubmed squality ; do
    for file in ./data/truncated_benchmarks/${data}/*.json ; do
      echo $data faithfulness $file
      python evaluation_tester.py \
        --dataset_path $file \
        --prompt_path ./prompts/${dim}.txt \
        --output_path ${save_dir}/${data}/${dim}/$(basename ${file}) \
        --model gpt-4-0613 \
        --dimension $dim \
        --temperature 0. \
        --n 1
    done
done
