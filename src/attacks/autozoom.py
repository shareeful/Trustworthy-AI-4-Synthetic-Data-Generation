"""
AutoZOOM adversarial attack implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class AutoZOOMAttack:
    """
    AutoZOOM: Autoencoder-based Zeroth Order Optimization Method.
    
    Black-box adversarial attack that uses gradient-free optimization
    to generate adversarial perturbations.
    
    Reference: Tu et al., "AutoZOOM: Autoencoder-based Zeroth Order 
    Optimization Method for Attacking Black-box Neural Networks", AAAI 2019.
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        max_iterations: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 64
    ):
        """
        Initialize AutoZOOM attack.
        
        Args:
            model: Target model to attack
            epsilon: Maximum perturbation magnitude
            max_iterations: Maximum optimization iterations
            learning_rate: Step size for gradient estimation
            batch_size: Batch size for attack
        """
        self.model = model
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        logger.info(f"Initialized AutoZOOM attack: ε={epsilon}, max_iter={max_iterations}")
    
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate adversarial examples using AutoZOOM.
        
        Args:
            x: Clean samples [batch_size, input_dim]
            y: True labels [batch_size]
            device: Computation device
            
        Returns:
            Tuple of (adversarial_samples, attack_success_rate)
        """
        self.model.eval()
        x = x.to(device)
        y = y.to(device)
        
        batch_size, input_dim = x.shape
        
        # Initialize adversarial samples as clean samples
        x_adv = x.clone().detach()
        
        # Initialize perturbations
        delta = torch.zeros_like(x, requires_grad=False)
        
        # Get original predictions
        with torch.no_grad():
            outputs = self.model(x)
            if isinstance(outputs, dict):
                original_preds = outputs['x_recon']
            else:
                original_preds = outputs
            
            threshold = original_preds.median()
            original_labels = (original_preds > threshold).float().squeeze()
        
        # Optimization loop
        success_count = 0
        for iteration in range(self.max_iterations):
            # Estimate gradient using finite differences
            grad_estimate = torch.zeros_like(x)
            
            for i in range(input_dim):
                # Create perturbation in i-th dimension
                e_i = torch.zeros_like(x)
                e_i[:, i] = self.learning_rate
                
                # Forward pass with positive perturbation
                x_plus = torch.clamp(x + delta + e_i, 0, 1)
                with torch.no_grad():
                    outputs_plus = self.model(x_plus)
                    if isinstance(outputs_plus, dict):
                        preds_plus = outputs_plus['x_recon']
                    else:
                        preds_plus = outputs_plus
                
                # Forward pass with negative perturbation
                x_minus = torch.clamp(x + delta - e_i, 0, 1)
                with torch.no_grad():
                    outputs_minus = self.model(x_minus)
                    if isinstance(outputs_minus, dict):
                        preds_minus = outputs_minus['x_recon']
                    else:
                        preds_minus = outputs_minus
                
                # Estimate gradient
                grad_estimate[:, i] = (preds_plus - preds_minus).squeeze() / (2 * self.learning_rate)
            
            # Update perturbation in direction that maximizes loss
            delta = delta + self.epsilon * torch.sign(grad_estimate) / self.max_iterations
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            
            # Apply perturbation
            x_adv = torch.clamp(x + delta, 0, 1)
            
            # Check success
            with torch.no_grad():
                outputs_adv = self.model(x_adv)
                if isinstance(outputs_adv, dict):
                    adv_preds = outputs_adv['x_recon']
                else:
                    adv_preds = outputs_adv
                
                adv_labels = (adv_preds > threshold).float().squeeze()
                success = (adv_labels != original_labels).float()
                success_count = success.sum().item()
            
            # Early stopping if all attacks succeeded
            if success_count == batch_size:
                logger.debug(f"AutoZOOM converged at iteration {iteration}")
                break
        
        attack_success_rate = success_count / batch_size
        logger.info(f"AutoZOOM attack success rate: {attack_success_rate:.2%}")
        
        return x_adv, attack_success_rate
    
    def generate_adversarial_batch(
        self,
        x_clean: torch.Tensor,
        y_clean: torch.Tensor,
        device: torch.device,
        adversarial_ratio: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mixed batch of clean and adversarial samples.
        
        Args:
            x_clean: Clean samples
            y_clean: Clean labels
            device: Computation device
            adversarial_ratio: Fraction of adversarial samples (e.g., 0.1 = 10%)
            
        Returns:
            Tuple of (mixed_batch, labels)
        """
        batch_size = x_clean.size(0)
        n_adversarial = int(batch_size * adversarial_ratio)
        
        if n_adversarial == 0:
            return x_clean, y_clean
        
        # Select random subset for adversarial attack
        adversarial_indices = np.random.choice(batch_size, n_adversarial, replace=False)
        clean_indices = np.array([i for i in range(batch_size) if i not in adversarial_indices])
        
        # Generate adversarial samples
        x_adv, _ = self.attack(
            x_clean[adversarial_indices],
            y_clean[adversarial_indices],
            device
        )
        
        # Combine clean and adversarial
        x_mixed = x_clean.clone()
        x_mixed[adversarial_indices] = x_adv
        
        logger.debug(f"Generated mixed batch: {len(clean_indices)} clean + {n_adversarial} adversarial")
        
        return x_mixed, y_clean


if __name__ == "__main__":
    # Test AutoZOOM attack
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(11, 11)
        
        def forward(self, x):
            return {'x_recon': torch.sigmoid(self.fc(x))}
    
    model = DummyModel()
    device = torch.device('cpu')
    
    # Create test data
    x_clean = torch.randn(32, 11)
    y_clean = torch.randint(0, 2, (32,)).float()
    
    # Initialize attack
    attack = AutoZOOMAttack(
        model=model,
        epsilon=0.1,
        max_iterations=50,
        learning_rate=0.01
    )
    
    # Generate adversarial samples
    print("\nGenerating adversarial samples...")
    x_adv, success_rate = attack.attack(x_clean, y_clean, device)
    
    print(f"\nResults:")
    print(f"  Clean samples shape: {x_clean.shape}")
    print(f"  Adversarial samples shape: {x_adv.shape}")
    print(f"  Attack success rate: {success_rate:.2%}")
    print(f"  Average perturbation: {(x_adv - x_clean).abs().mean().item():.4f}")
    
    # Test mixed batch generation
    print("\n\nGenerating mixed batch...")
    x_mixed, y_mixed = attack.generate_adversarial_batch(
        x_clean, y_clean, device, adversarial_ratio=0.1
    )
    print(f"  Mixed batch shape: {x_mixed.shape}")