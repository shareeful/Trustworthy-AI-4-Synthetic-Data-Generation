from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class RegulatoryProfile:
    name: str
    article: str
    dimension: str
    priority_vector: np.ndarray
    thresholds: Dict[str, float]
    alpha: float = 5.0
    beta: float = 0.5

    def translate_weights(self) -> Dict[str, float]:
        p = self.priority_vector
        return {
            "recon": 1.0,
            "fair":  float(self.alpha * p[0] + self.beta),
            "adv":   float(self.alpha * p[1] + self.beta),
            "sparse":float(self.alpha * p[2] + self.beta),
        }


P1_HIGH_RISK_MEDICAL = RegulatoryProfile(
    name="P-1: High-Risk Medical",
    article="Article 6 (Non-discrimination)",
    dimension="Fairness",
    priority_vector=np.array([0.6, 0.3, 0.1]),
    thresholds={
        "equalized_odds": 0.05,
        "demographic_parity": 0.05,
        "utility_floor": 0.60,
    },
)

P2_SAFETY_CRITICAL = RegulatoryProfile(
    name="P-2: Safety-Critical",
    article="Article 15 (Robustness)",
    dimension="Security",
    priority_vector=np.array([0.2, 0.7, 0.1]),
    thresholds={
        "attack_success_rate": 0.15,
        "clean_accuracy": 0.60,
        "utility_floor": 0.60,
    },
)

P3_AUDIT_READY = RegulatoryProfile(
    name="P-3: Audit-Ready",
    article="Article 13 (Transparency)",
    dimension="Explainability",
    priority_vector=np.array([0.1, 0.2, 0.7]),
    thresholds={
        "shap_stability": 0.90,
        "sparsity": 0.10,
        "utility_floor": 0.60,
    },
)

PROFILES = {
    "P1": P1_HIGH_RISK_MEDICAL,
    "P2": P2_SAFETY_CRITICAL,
    "P3": P3_AUDIT_READY,
}
