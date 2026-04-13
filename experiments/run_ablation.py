import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from models.profiles import P1_HIGH_RISK_MEDICAL
from models.vae_gan import VAEGAN
from data.preprocessing import preprocess, baseline_preprocessing
from engine.compliance_engine import ClosedLoopComplianceEngine
from utils.metrics import equalized_odds, utility


def eval_fn(model, X, sensitive):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_syn = model.sample(len(X), device).cpu().numpy()
    clf = LogisticRegression(max_iter=500, random_state=42)
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X_syn, (X_syn[:, 0] > 0.5).astype(int), sensitive,
        test_size=0.2, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    eq  = equalized_odds(y_te, y_pred, s_te)
    acc = utility(y_te, y_pred)
    return {"equalized_odds": eq, "utility": acc}


def run_config(X, sensitive, weights, use_ipw, use_translation,
               use_engine, device, seed, output_dir):
    np.random.seed(seed)
    torch.manual_seed(seed)

    profile = P1_HIGH_RISK_MEDICAL
    model   = VAEGAN(input_dim=X.shape[1], latent_dim=16)

    if not use_translation:
        fixed_weights = {"recon": 1.0, "fair": 0.5, "adv": 0.5, "sparse": 0.5}
    else:
        fixed_weights = None

    engine = ClosedLoopComplianceEngine(
        model=model,
        profile=profile,
        device=device,
        eval_interval=10,
        delta=0.1 if use_engine else 0.0,
        max_epochs=300,
        batch_size=64,
        lr=2e-4,
    )

    if fixed_weights is not None:
        engine.weights = fixed_weights

    sw = weights if use_ipw else np.ones(len(X))
    result = engine.train(X, sensitive, sample_weights=sw,
                          eval_fn=lambda m, x, s: eval_fn(m, x, s))
    return result


def run(data_path: str, output_dir: str = "results", seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    profile = P1_HIGH_RISK_MEDICAL

    sensitive_col = "gender" if "gender" in df.columns else df.columns[1]
    df_proc, weights = preprocess(df, profile, sensitive_col=sensitive_col)
    sensitive = (df[sensitive_col].values == df[sensitive_col].max()).astype(float) \
        if sensitive_col in df.columns else np.zeros(len(df))
    X = df_proc.values.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        ("Baseline",          False, False, False),
        ("+Phase2_IPW",       True,  False, False),
        ("+Phase3_Trans",     True,  True,  False),
        ("+Engine_Loop",      False, False, True),
        ("Full_Method",       True,  True,  True),
    ]

    results = {}
    for name, use_ipw, use_trans, use_eng in configs:
        print(f"Running ablation config: {name}")
        r = run_config(X, sensitive, weights, use_ipw, use_trans,
                       use_eng, device, seed, output_dir)
        results[name] = {
            "equalized_odds": r["final_metrics"].get("equalized_odds", None),
            "adjustment_cycles": r["adjustment_cycles"],
            "training_time_min": r["training_time_min"],
            "compliant": r["compliant"],
        }
        print(f"  Eq.Odds={results[name]['equalized_odds']} "
              f"Cycles={results[name]['adjustment_cycles']} "
              f"Compliant={results[name]['compliant']}")

    out_path = os.path.join(output_dir, f"ablation_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAblation results saved to {out_path}")
    return results


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/insurance.csv"
    run(data_path)
