import json
import os

from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau
from tqdm import tqdm


def calculate_correlation(pred_score, human_score):
    assert len(pred_score) == len(human_score)
    result = {
        "pearson": round(pearsonr(pred_score, human_score)[0], 4),
        "spearman": round(spearmanr(pred_score, human_score)[0], 4),
        "kendalltau": round(kendalltau(pred_score, human_score)[0], 4),
    }
    return result


def print_correlations(result):
    table = PrettyTable(["Pearson", "Spearman", "Kendall"])
    table.add_row([result[key] for key in ("pearson", "spearman", "kendalltau")])
    print(table)


class CorrelationCalculator:
    def __init__(self, dimension: str):
        self.dimension = dimension

    def correlation_output(self, input_fp: str):
        input_eval = json.load(open(input_fp))

        pred_scores, human_scores = [], []
        print(f"Calculating correlations with respect to {self.dimension}...")

        for instance in tqdm(input_eval, dynamic_ncols=True):
            pred_scores.append(instance["model_score"])
            if self.dimension == "consistency":  # TODO: No need
                human_scores.append(instance["scores"]["factual_consistency"])
            else:
                human_scores.append(instance["scores"][self.dimension])

        results = calculate_correlation(pred_scores, human_scores)
        print_correlations(results)
        results["dimension"] = self.dimension
        results["dataset"] = os.path.basename(input_fp)

        return results
