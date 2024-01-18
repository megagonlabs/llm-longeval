import json
import os

import fire

from eval.correlation import CorrelationCalculator
from eval.evaluator import Evaluator


def run(
    dataset_path: str,
    prompt_path: str,
    output_path: str,
    model: str,
    dimension: str,
    temperature: float = 0.0,
    n: int = 1,
    api_key: str = None,
):
    # Evaluation
    print("Evaluating... ")
    test_evaluator = Evaluator(model=model, api_key=api_key)
    eval_results = test_evaluator.eval_output(
        dataset_path, prompt_path, temperature=temperature, n=n
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_results, f)

    # Correlation calculation
    print(f"Calculating Correlation...")
    corr_calculator = CorrelationCalculator(dimension=dimension)
    corr_results = corr_calculator.correlation_output(output_path)

    # save the correlation results into file
    save_corr_path = output_path.replace(".json", "_corr.json")
    with open(save_corr_path, "w") as f:
        json.dump(corr_results, f)


if __name__ == "__main__":
    fire.Fire(run)
