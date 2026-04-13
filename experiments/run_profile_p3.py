import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from models.profiles import P2_SAFETY_CRITICAL
from models.vae_gan import VAEGAN
from data.preprocessing import preprocess
from engine.compliance_engine import ClosedLoopComplianceEngine
from utils.metrics import jensen_shannon_divergence, utility


def simulate_attack_success(model, X, device, eps=0.1, n_samples=200):
    model.eval()
    X_t = torch.FloatTensor(X[:n_samples]).to(device)
    with torch.no_grad():
        X_syn = model.sample(n_samples, device)

    noise = torch.randn_like(X_syn) * eps
    X_adv = torch.clamp(X_syn + noise, 0.0, 1.0)

    disc_clean = model.discriminator(X_syn)
    disc_adv   = model.discriminator(X_adv)

    fooled = (disc_adv < 0.5).float().mean().item()
    return fooled


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

    asr = simulate_attack_success(model, X, device)
    acc = utility(y_te, y_pred)
    jsd = jensen_shannon_divergence(X, X_syn)

    return {
        "attack_success_rate": asr,
        "clean_accuracy": acc,
        "utility": acc,
        "jsd": jsd,
    }


def run(data_path: str, output_dir: str = "results", seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    profile = P2_SAFETY_CRITICAL

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
        adv_batch_fraction=0.10,
    )

    result = engine.train(X, sensitive, sample_weights=weights,
                          eval_fn=lambda m, x, s: eval_fn(m, x, s))

    out_path = os.path.join(output_dir, f"p2_result_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    engine.save_traceability_log(
        os.path.join(output_dir, f"p2_traceability_seed{seed}.json"))

    print(f"Profile P-2 | Attack Rate: {result['final_metrics'].get('attack_success_rate', 'N/A'):.4f} "
          f"| Clean Acc: {result['final_metrics'].get('clean_accuracy', 'N/A'):.4f} "
          f"| Cycles: {result['adjustment_cycles']} "
          f"| Compliant: {result['compliant']}")
    return result


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/insurance.csv"
    run(data_path)
