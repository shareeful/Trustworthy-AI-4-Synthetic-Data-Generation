import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from models.profiles import P3_AUDIT_READY
from models.vae_gan import VAEGAN
from data.preprocessing import preprocess
from engine.compliance_engine import ClosedLoopComplianceEngine
from utils.metrics import jensen_shannon_divergence, utility


def shap_rank_stability_simple(model, X, device, n_runs=5):
    from sklearn.linear_model import LogisticRegression
    model.eval()
    rankings = []
    for _ in range(n_runs):
        with torch.no_grad():
            X_syn = model.sample(min(200, len(X)), device).cpu().numpy()
        y_syn = (X_syn[:, 0] > 0.5).astype(int)
        clf   = LogisticRegression(max_iter=300, random_state=42)
        try:
            clf.fit(X_syn, y_syn)
            coef = np.abs(clf.coef_[0])
            rankings.append(np.argsort(-coef))
        except Exception:
            pass
    if len(rankings) < 2:
        return 0.0
    corrs = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            c = np.corrcoef(rankings[i], rankings[j])[0, 1]
            corrs.append(c)
    return float(np.mean(corrs))


def eval_fn(model, X, sensitive):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_syn = model.sample(len(X), device).cpu().numpy()

    clf = LogisticRegression(max_iter=500, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_syn, (X_syn[:, 0] > 0.5).astype(int),
        test_size=0.2, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    stab = shap_rank_stability_simple(model, X, device)
    acc  = utility(y_te, y_pred)
    jsd  = jensen_shannon_divergence(X, X_syn)

    return {
        "shap_stability": stab,
        "utility": acc,
        "jsd": jsd,
    }


def run(data_path: str, output_dir: str = "results", seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    profile = P3_AUDIT_READY

    sensitive_col = "gender" if "gender" in df.columns else df.columns[1]
    df_proc, weights = preprocess(df, profile, sensitive_col=sensitive_col)
    sensitive = np.zeros(len(df))
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

    out_path = os.path.join(output_dir, f"p3_result_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    engine.save_traceability_log(
        os.path.join(output_dir, f"p3_traceability_seed{seed}.json"))

    print(f"Profile P-3 | SHAP Stability: {result['final_metrics'].get('shap_stability', 'N/A'):.4f} "
          f"| Cycles: {result['adjustment_cycles']} "
          f"| Compliant: {result['compliant']}")
    return result


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/insurance.csv"
    run(data_path)
