import torch
import torch.nn.functional as F


def reconstruction_loss(x: torch.Tensor,
                        x_hat: torch.Tensor,
                        mu: torch.Tensor,
                        log_var: torch.Tensor) -> torch.Tensor:
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + kl


def fairness_loss(y_hat: torch.Tensor,
                  sensitive: torch.Tensor) -> torch.Tensor:
    groups = sensitive.unique()
    if len(groups) < 2:
        return torch.tensor(0.0, requires_grad=True)
    rates = []
    for g in groups:
        mask = sensitive == g
        if mask.sum() == 0:
            continue
        rates.append(y_hat[mask].mean())
    if len(rates) < 2:
        return torch.tensor(0.0, requires_grad=True)
    return torch.abs(rates[0] - rates[1])


def adversarial_loss(real_pred: torch.Tensor,
                     fake_pred: torch.Tensor) -> torch.Tensor:
    real_loss = F.binary_cross_entropy(real_pred,
                                        torch.ones_like(real_pred))
    fake_loss = F.binary_cross_entropy(fake_pred,
                                        torch.zeros_like(fake_pred))
    return real_loss + fake_loss


def sparsity_loss(z: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(z))


def total_loss(recon: torch.Tensor,
               fair: torch.Tensor,
               adv: torch.Tensor,
               sparse: torch.Tensor,
               weights: dict) -> torch.Tensor:
    return (weights["recon"]  * recon  +
            weights["fair"]   * fair   +
            weights["adv"]    * adv    +
            weights["sparse"] * sparse)
