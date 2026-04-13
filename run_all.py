import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
from experiments.run_profile_p1 import run as run_p1
from experiments.run_profile_p3 import run as run_p2
from experiments.run_profile_p3_exp import run as run_p3
from experiments.run_ablation import run as run_ablation


SEEDS = [42, 123, 456, 789, 1024]


def aggregate(results: list, key: str):
    vals = [r["final_metrics"].get(key, None) for r in results
            if r["final_metrics"].get(key) is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def run_all(data_path: str, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)

    p1_results, p2_results, p3_results = [], [], []

    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"Seed {seed}")
        print(f"{'='*50}")
        p1_results.append(run_p1(data_path, output_dir, seed))
        p2_results.append(run_p2(data_path, output_dir, seed))
        p3_results.append(run_p3(data_path, output_dir, seed))

    run_ablation(data_path, output_dir, seed=42)

    summary = {
        "P1_Fairness": {
            "equalized_odds_mean": aggregate(p1_results, "equalized_odds")[0],
            "equalized_odds_std":  aggregate(p1_results, "equalized_odds")[1],
            "jsd_mean":            aggregate(p1_results, "jsd")[0],
            "utility_mean":        aggregate(p1_results, "utility")[0],
        },
        "P2_Security": {
            "attack_rate_mean":    aggregate(p2_results, "attack_success_rate")[0],
            "attack_rate_std":     aggregate(p2_results, "attack_success_rate")[1],
            "clean_acc_mean":      aggregate(p2_results, "clean_accuracy")[0],
        },
        "P3_Explainability": {
            "shap_stability_mean": aggregate(p3_results, "shap_stability")[0],
            "shap_stability_std":  aggregate(p3_results, "shap_stability")[1],
            "utility_mean":        aggregate(p3_results, "utility")[0],
        },
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(json.dumps(summary, indent=2))
    print(f"\nFull results saved to {output_dir}/")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/insurance.csv"
    run_all(data_path)
