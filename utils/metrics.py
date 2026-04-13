import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import jensenshannon
import shap
from typing import Dict, Any


def equalized_odds(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   sensitive: np.ndarray) -> float:
    groups = np.unique(sensitive)
    tprs = []
    for g in groups:
        mask = sensitive == g
        pos_mask = (y_true[mask] == 1)
        if pos_mask.sum() == 0:
            continue
        tpr = (y_pred[mask][pos_mask] == 1).mean()
        tprs.append(tpr)
    if len(tprs) < 2:
        return 0.0
    return float(abs(tprs[0] - tprs[1]))


def demographic_parity(y_pred: np.ndarray,
                       sensitive: np.ndarray) -> float:
    groups = np.unique(sensitive)
    rates = []
    for g in groups:
        mask = sensitive == g
        rates.append(y_pred[mask].mean())
    if len(rates) < 2:
        return 0.0
    return float(abs(rates[0] - rates[1]))


def jensen_shannon_divergence(real: np.ndarray,
                               synthetic: np.ndarray,
                               bins: int = 50) -> float:
    all_vals = np.concatenate([real.flatten(), synthetic.flatten()])
    min_v, max_v = all_vals.min(), all_vals.max()
    edges = np.linspace(min_v, max_v, bins + 1)
    p, _ = np.histogram(real.flatten(), bins=edges, density=True)
    q, _ = np.histogram(synthetic.flatten(), bins=edges, density=True)
    p = p + 1e-10
    q = q + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))


def shap_rank_stability(model: Any,
                         X: np.ndarray,
                         n_runs: int = 5) -> float:
    explainer = shap.KernelExplainer(model.predict_proba,
                                      shap.sample(X, 50))
    rankings = []
    for _ in range(n_runs):
        idx = np.random.choice(len(X), min(50, len(X)), replace=False)
        sv  = explainer.shap_values(X[idx], silent=True)
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        rankings.append(np.argsort(-mean_abs))
    correlations = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            r = np.corrcoef(rankings[i], rankings[j])[0, 1]
            correlations.append(r)
    return float(np.mean(correlations)) if correlations else 0.0


def utility(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_all_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         sensitive: np.ndarray,
                         real_data: np.ndarray,
                         synthetic_data: np.ndarray) -> Dict[str, float]:
    return {
        "equalized_odds":     equalized_odds(y_true, y_pred, sensitive),
        "demographic_parity": demographic_parity(y_pred, sensitive),
        "jsd":                jensen_shannon_divergence(real_data, synthetic_data),
        "utility":            utility(y_true, y_pred),
    }
