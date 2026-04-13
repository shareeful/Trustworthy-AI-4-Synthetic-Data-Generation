import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.profiles import RegulatoryProfile
from models.losses import (reconstruction_loss, fairness_loss,
                            adversarial_loss, sparsity_loss, total_loss)
from models.vae_gan import VAEGAN


@dataclass
class TraceabilityEntry:
    epoch: int
    weights: Dict[str, float]
    metrics: Dict[str, float]
    violations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ClosedLoopComplianceEngine:
    def __init__(self,
                 model: VAEGAN,
                 profile: RegulatoryProfile,
                 device: torch.device,
                 eval_interval: int = 10,
                 delta: float = 0.1,
                 max_epochs: int = 300,
                 batch_size: int = 64,
                 lr: float = 2e-4,
                 adv_batch_fraction: float = 0.10):
        self.model     = model.to(device)
        self.profile   = profile
        self.device    = device
        self.interval  = eval_interval
        self.delta     = delta
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.adv_frac  = adv_batch_fraction

        self.weights   = profile.translate_weights()
        self.log: List[TraceabilityEntry] = []

        self.opt_enc_gen = optim.Adam(
            list(model.encoder.parameters()) +
            list(model.generator.parameters()),
            lr=lr, betas=(0.5, 0.999))
        self.opt_disc = optim.Adam(
            model.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.999))

    def _add_adversarial_noise(self, x: torch.Tensor,
                                eps: float = 0.1) -> torch.Tensor:
        noise = torch.randn_like(x) * eps
        return torch.clamp(x + noise, 0.0, 1.0)

    def _train_step(self, x: torch.Tensor,
                    sensitive: torch.Tensor,
                    sample_weights: torch.Tensor):
        p2 = "P-2" in self.profile.name
        if p2:
            n_adv = max(1, int(self.adv_frac * len(x)))
            idx   = torch.randperm(len(x))[:n_adv]
            x_aug = x.clone()
            x_aug[idx] = self._add_adversarial_noise(x[idx])
        else:
            x_aug = x

        x_hat, mu, log_var, z = self.model(x_aug)

        real_pred = self.model.discriminator(x)
        fake_pred = self.model.discriminator(x_hat.detach())
        d_loss = adversarial_loss(real_pred, fake_pred)

        self.opt_disc.zero_grad()
        d_loss.backward()
        self.opt_disc.step()

        x_hat, mu, log_var, z = self.model(x_aug)
        fake_pred_g = self.model.discriminator(x_hat)

        l_recon  = reconstruction_loss(x, x_hat, mu, log_var)
        l_fair   = fairness_loss(x_hat[:, 0], sensitive)
        l_adv    = -torch.mean(torch.log(fake_pred_g + 1e-8))
        l_sparse = sparsity_loss(z)

        loss = total_loss(l_recon, l_fair, l_adv, l_sparse, self.weights)
        weighted_loss = (loss * sample_weights).mean()

        self.opt_enc_gen.zero_grad()
        weighted_loss.backward()
        self.opt_enc_gen.step()

        return {
            "recon": l_recon.item(),
            "fair": l_fair.item(),
            "adv": l_adv.item(),
            "sparse": l_sparse.item(),
            "total": weighted_loss.item(),
        }

    def _check_violations(self, metrics: Dict[str, float]) -> List[str]:
        v = []
        t = self.profile.thresholds
        if "equalized_odds" in t and metrics.get("equalized_odds", 1.0) > t["equalized_odds"]:
            v.append("fairness")
        if "attack_success_rate" in t and metrics.get("attack_success_rate", 1.0) > t["attack_success_rate"]:
            v.append("security")
        if "shap_stability" in t and metrics.get("shap_stability", 0.0) < t["shap_stability"]:
            v.append("explainability")
        return v

    def _update_weights(self, violations: List[str]):
        if "fairness" in violations:
            self.weights["fair"] += self.delta
        if "security" in violations:
            self.weights["adv"]  += self.delta
        if "explainability" in violations:
            self.weights["sparse"] += self.delta

    def _all_satisfied(self, metrics: Dict[str, float]) -> bool:
        return len(self._check_violations(metrics)) == 0

    def train(self,
              X: np.ndarray,
              sensitive: np.ndarray,
              sample_weights: Optional[np.ndarray] = None,
              eval_fn=None) -> Dict:

        if sample_weights is None:
            sample_weights = np.ones(len(X))

        X_t  = torch.FloatTensor(X).to(self.device)
        s_t  = torch.FloatTensor(sensitive).to(self.device)
        w_t  = torch.FloatTensor(sample_weights).to(self.device)

        dataset = TensorDataset(X_t, s_t, w_t)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        start = time.time()
        adjustment_cycles = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for xb, sb, wb in loader:
                self._train_step(xb, sb, wb)

            if epoch % self.interval == 0:
                self.model.eval()
                metrics = eval_fn(self.model, X, sensitive) if eval_fn else {}

                violations = self._check_violations(metrics)
                if violations:
                    self._update_weights(violations)
                    adjustment_cycles += 1

                entry = TraceabilityEntry(
                    epoch=epoch,
                    weights=dict(self.weights),
                    metrics=metrics,
                    violations=violations,
                )
                self.log.append(entry)

                if metrics.get("utility", 1.0) < self.profile.thresholds.get("utility_floor", 0.60):
                    break
                if self._all_satisfied(metrics):
                    break

        elapsed = time.time() - start
        return {
            "profile": self.profile.name,
            "article": self.profile.article,
            "initial_weights": self.profile.translate_weights(),
            "final_weights": dict(self.weights),
            "adjustment_cycles": adjustment_cycles,
            "training_time_min": round(elapsed / 60, 2),
            "final_metrics": self.log[-1].metrics if self.log else {},
            "compliant": self._all_satisfied(self.log[-1].metrics if self.log else {}),
            "log": [vars(e) for e in self.log],
        }

    def save_traceability_log(self, path: str):
        with open(path, "w") as f:
            json.dump([vars(e) for e in self.log], f, indent=2, default=str)
