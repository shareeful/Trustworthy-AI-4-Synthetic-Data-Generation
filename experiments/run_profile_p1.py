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
from data.preprocessing import preprocess
from engine.compliance_engine import ClosedLoopComplianceEngine
from utils.metrics import equalized_odds, jensen_shannon_divergence, utility


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def eval_fn(model, X, sensitive):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t   = torch.FloatTensor(X).to(device)
        X_syn = model.sample(len(X), device).cpu().numpy()

    clf = LogisticRegression(max_iter=500, random_state=42)
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X_syn, (X_syn[:, 0] > 0.5).astype(int), sensitive,
        test_size=0.2, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_true = y_te

    eq = equalized_odds(y_true, y_pred, s_te)
    jsd = jensen_shannon_divergence(X, X_syn)
    acc = utility(y_true, y_pred)

    return {
        "equalized_odds": eq,
        "jsd": jsd,
        "utility": acc,
    }


def run(data_path: str, output_dir: str = "results", seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_path)
    profile = P1_HIGH_RISK_MEDICAL

    sensitive_col = "gender" if "gender" in df.columns else df.columns[1]
    df_proc, weights = preprocess(df, profile, sensitive_col=sensitive_col)

    sensitive = df[sensitive_col].values if sensitive_col in df.columns else np.zeros(len(df))
    sensitive = (sensitive == sensitive.max()).astype(float)

    X = df_proc.values.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = VAEGAN(input_dim=X.shape[1], latent_dim=16)

    engine = ClosedLoopComplianceEngine(
        model=model,
        profile=profile,
        device=device,
        eval_interval=10,
        delta=0.1,
        max_epochs=300,
        batch_size=64,
        lr=2e-4,
    )

    result = engine.train(X, sensitive, sample_weights=weights,
                          eval_fn=lambda m, x, s: eval_fn(m, x, s))

    out_path = os.path.join(output_dir, f"p1_result_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log_path = os.path.join(output_dir, f"p1_traceability_seed{seed}.json")
    engine.save_traceability_log(log_path)

    print(f"Profile P-1 | Equalized Odds: {result['final_metrics'].get('equalized_odds', 'N/A'):.4f} "
          f"| Cycles: {result['adjustment_cycles']} "
          f"| Time: {result['training_time_min']} min "
          f"| Compliant: {result['compliant']}")
    return result


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/insurance.csv"
    run(data_path)
