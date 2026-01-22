"""
Trustworthy AI Metrics: Fairness, Security, Explainability.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FairnessMetrics:
    """Compute fairness metrics for demographic groups."""
    
    @staticmethod
    def equalized_odds(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> float:
        """
        Compute Equalized Odds Difference.
        
        Measures: |P(ŷ=1|y=1,s=0) - P(ŷ=1|y=1,s=1)|
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attr: Protected attribute (e.g., gender)
            
        Returns:
            Equalized odds difference (lower is better, 0 is perfect)
        """
        groups = np.unique(sensitive_attr)
        if len(groups) != 2:
            logger.warning(f"Equalized Odds expects binary sensitive attribute, got {len(groups)} groups")
        
        # Filter to positive class (y=1)
        pos_mask = (y_true == 1)
        
        tprs = []
        for group in groups:
            group_mask = (sensitive_attr == group) & pos_mask
            if group_mask.sum() == 0:
                logger.warning(f"No positive samples for group {group}")
                continue
            
            tpr = (y_pred[group_mask] == 1).mean()
            tprs.append(tpr)
        
        if len(tprs) < 2:
            return 0.0
        
        eq_odds_diff = abs(tprs[0] - tprs[1])
        return eq_odds_diff
    
    @staticmethod
    def demographic_parity(
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> float:
        """
        Compute Demographic Parity Difference.
        
        Measures: |P(ŷ=1|s=0) - P(ŷ=1|s=1)|
        
        Args:
            y_pred: Predicted labels
            sensitive_attr: Protected attribute
            
        Returns:
            Demographic parity difference (lower is better)
        """
        groups = np.unique(sensitive_attr)
        
        pred_rates = []
        for group in groups:
            group_mask = (sensitive_attr == group)
            pred_rate = (y_pred[group_mask] == 1).mean()
            pred_rates.append(pred_rate)
        
        if len(pred_rates) < 2:
            return 0.0
        
        dp_diff = abs(pred_rates[0] - pred_rates[1])
        return dp_diff
    
    @staticmethod
    def compute_bias_loss(
        predictions: torch.Tensor,
        sensitive_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute differentiable bias loss for training.
        
        L_Bias = |P(ŷ=1|s=0) - P(ŷ=1|s=1)|
        
        Args:
            predictions: Model predictions [batch_size]
            sensitive_attr: Protected attribute [batch_size]
            
        Returns:
            Bias loss tensor
        """
        groups = torch.unique(sensitive_attr)
        
        if len(groups) != 2:
            return torch.tensor(0.0, device=predictions.device)
        
        group_0_mask = (sensitive_attr == groups[0])
        group_1_mask = (sensitive_attr == groups[1])
        
        rate_0 = predictions[group_0_mask].mean()
        rate_1 = predictions[group_1_mask].mean()
        
        bias_loss = torch.abs(rate_0 - rate_1)
        return bias_loss


class SecurityMetrics:
    """Compute security/robustness metrics."""
    
    @staticmethod
    def attack_success_rate(
        model: nn.Module,
        x_adv: torch.Tensor,
        y_true: torch.Tensor,
        device: torch.device
    ) -> float:
        """
        Compute attack success rate.
        
        ASR = fraction of adversarial samples that changed prediction
        
        Args:
            model: Trained model
            x_adv: Adversarial samples
            y_true: True labels
            device: Computation device
            
        Returns:
            Attack success rate (lower is better)
        """
        model.eval()
        with torch.no_grad():
            x_adv = x_adv.to(device)
            y_true = y_true.to(device)
            
            # Get predictions on adversarial samples
            outputs = model(x_adv)
            if isinstance(outputs, dict):
                predictions = outputs['x_recon']
            else:
                predictions = outputs
            
            # Convert to binary predictions (assuming regression task)
            # Threshold at median
            threshold = predictions.median()
            y_pred_adv = (predictions > threshold).float().squeeze()
            
            # Success = predictions changed from true labels
            success = (y_pred_adv != y_true).float().mean().item()
        
        return success
    
    @staticmethod
    def clean_accuracy(
        model: nn.Module,
        x_clean: torch.Tensor,
        y_true: torch.Tensor,
        device: torch.device
    ) -> float:
        """
        Compute accuracy on clean (non-adversarial) samples.
        
        Args:
            model: Trained model
            x_clean: Clean samples
            y_true: True labels
            device: Computation device
            
        Returns:
            Accuracy on clean samples
        """
        model.eval()
        with torch.no_grad():
            x_clean = x_clean.to(device)
            y_true = y_true.to(device)
            
            outputs = model(x_clean)
            if isinstance(outputs, dict):
                predictions = outputs['x_recon']
            else:
                predictions = outputs
            
            threshold = predictions.median()
            y_pred = (predictions > threshold).float().squeeze()
            
            accuracy = (y_pred == y_true).float().mean().item()
        
        return accuracy


class ExplainabilityMetrics:
    """Compute explainability/interpretability metrics."""
    
    @staticmethod
    def shap_rank_stability(
        shap_values_1: np.ndarray,
        shap_values_2: np.ndarray,
        top_k: int = 5
    ) -> float:
        """
        Compute SHAP rank stability between two sets of explanations.
        
        Measures consistency of feature importance rankings.
        
        Args:
            shap_values_1: First set of SHAP values [n_features]
            shap_values_2: Second set of SHAP values [n_features]
            top_k: Number of top features to compare
            
        Returns:
            Rank stability (higher is better, 1.0 is perfect)
        """
        # Get top-k feature indices for each
        top_k_1 = np.argsort(np.abs(shap_values_1))[-top_k:]
        top_k_2 = np.argsort(np.abs(shap_values_2))[-top_k:]
        
        # Compute Jaccard similarity
        intersection = len(set(top_k_1) & set(top_k_2))
        union = len(set(top_k_1) | set(top_k_2))
        
        stability = intersection / union if union > 0 else 0.0
        return stability
    
    @staticmethod
    def feature_sparsity(shap_values: np.ndarray, threshold: float = 0.01) -> float:
        """
        Compute feature sparsity ratio.
        
        Measures fraction of features with negligible importance.
        
        Args:
            shap_values: SHAP importance values [n_features]
            threshold: Threshold for considering feature as spurious
            
        Returns:
            Sparsity ratio (fraction of features below threshold)
        """
        normalized_shap = np.abs(shap_values) / (np.abs(shap_values).sum() + 1e-8)
        sparse_ratio = (normalized_shap < threshold).mean()
        return sparse_ratio
    
    @staticmethod
    def compute_sparsity_loss(latent_codes: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 sparsity loss on latent codes.
        
        L_L1 = E[|z|]
        
        Args:
            latent_codes: Latent representations [batch_size, latent_dim]
            
        Returns:
            L1 sparsity loss
        """
        return torch.mean(torch.abs(latent_codes))
    
    @staticmethod
    def identify_spurious_features(
        feature_names: list,
        shap_values: np.ndarray,
        spurious_keywords: list = ['id', 'index', 'patient']
    ) -> Dict[str, float]:
        """
        Identify spurious features based on naming and importance.
        
        Args:
            feature_names: List of feature names
            shap_values: SHAP importance values
            spurious_keywords: Keywords indicating spurious features
            
        Returns:
            Dictionary of {feature_name: importance} for spurious features
        """
        spurious_features = {}
        
        for i, name in enumerate(feature_names):
            # Check if feature name contains spurious keywords
            is_spurious = any(keyword in name.lower() for keyword in spurious_keywords)
            
            if is_spurious:
                spurious_features[name] = float(shap_values[i])
        
        return spurious_features


class MetricsAggregator:
    """Aggregate all T-AI metrics for compliance monitoring."""
    
    def __init__(self):
        self.fairness = FairnessMetrics()
        self.security = SecurityMetrics()
        self.explainability = ExplainabilityMetrics()
        
        logger.info("Initialized MetricsAggregator for T-AI compliance monitoring")
    
    def compute_all_metrics(
        self,
        model: nn.Module,
        x_real: torch.Tensor,
        y_true: torch.Tensor,
        sensitive_attr: np.ndarray,
        x_adv: Optional[torch.Tensor] = None,
        shap_values: Optional[np.ndarray] = None,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, float]:
        """
        Compute all T-AI metrics.
        
        Args:
            model: Trained model
            x_real: Real data samples
            y_true: True labels
            sensitive_attr: Protected attribute values
            x_adv: Adversarial samples (optional, for security)
            shap_values: SHAP values (optional, for explainability)
            device: Computation device
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Get model predictions for fairness
        model.eval()
        with torch.no_grad():
            outputs = model(x_real.to(device))
            if isinstance(outputs, dict):
                predictions = outputs['x_recon']
            else:
                predictions = outputs
            
            # Convert to binary predictions
            threshold = predictions.median()
            y_pred = (predictions > threshold).float().squeeze().cpu().numpy()
        
        # Fairness metrics
        try:
            metrics['equalized_odds'] = self.fairness.equalized_odds(
                y_true.cpu().numpy(), y_pred, sensitive_attr
            )
            metrics['demographic_parity'] = self.fairness.demographic_parity(
                y_pred, sensitive_attr
            )
        except Exception as e:
            logger.warning(f"Failed to compute fairness metrics: {e}")
            metrics['equalized_odds'] = 1.0  # Worst case
            metrics['demographic_parity'] = 1.0
        
        # Security metrics
        if x_adv is not None:
            try:
                metrics['attack_success_rate'] = self.security.attack_success_rate(
                    model, x_adv, y_true, device
                )
                metrics['clean_accuracy'] = self.security.clean_accuracy(
                    model, x_real, y_true, device
                )
            except Exception as e:
                logger.warning(f"Failed to compute security metrics: {e}")
                metrics['attack_success_rate'] = 1.0
                metrics['clean_accuracy'] = 0.0
        
        # Explainability metrics
        if shap_values is not None:
            try:
                # For rank stability, compare with baseline (uniform importance)
                baseline_shap = np.ones_like(shap_values) / len(shap_values)
                metrics['shap_rank_stability'] = self.explainability.shap_rank_stability(
                    shap_values, baseline_shap
                )
                metrics['feature_sparsity'] = self.explainability.feature_sparsity(shap_values)
            except Exception as e:
                logger.warning(f"Failed to compute explainability metrics: {e}")
                metrics['shap_rank_stability'] = 0.0
                metrics['feature_sparsity'] = 0.0
        
        return metrics
    
    def check_compliance(
        self,
        metrics: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if metrics satisfy compliance thresholds.
        
        Args:
            metrics: Computed metrics
            thresholds: Threshold dictionary from profile
            
        Returns:
            Tuple of (all_compliant, individual_compliance)
        """
        compliance = {}
        
        # Check each threshold
        if 'equalized_odds' in thresholds and 'equalized_odds' in metrics:
            compliance['equalized_odds'] = metrics['equalized_odds'] < thresholds['equalized_odds']
        
        if 'demographic_parity' in thresholds and 'demographic_parity' in metrics:
            compliance['demographic_parity'] = metrics['demographic_parity'] < thresholds['demographic_parity']
        
        if 'attack_success_rate' in thresholds and 'attack_success_rate' in metrics:
            compliance['attack_success_rate'] = metrics['attack_success_rate'] < thresholds['attack_success_rate']
        
        if 'clean_accuracy' in thresholds and 'clean_accuracy' in metrics:
            compliance['clean_accuracy'] = metrics['clean_accuracy'] > thresholds['clean_accuracy']
        
        if 'shap_rank_stability' in thresholds and 'shap_rank_stability' in metrics:
            compliance['shap_rank_stability'] = metrics['shap_rank_stability'] > thresholds['shap_rank_stability']
        
        if 'feature_sparsity' in thresholds and 'feature_sparsity' in metrics:
            compliance['feature_sparsity'] = metrics['feature_sparsity'] > thresholds['feature_sparsity']
        
        # Overall compliance
        all_compliant = all(compliance.values()) if compliance else False
        
        return all_compliant, compliance


if __name__ == "__main__":
    # Test metrics
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = y_true.copy()
    y_pred[np.random.rand(n_samples) < 0.1] = 1 - y_pred[np.random.rand(n_samples) < 0.1]  # 10% error
    
    sensitive_attr = np.random.choice(['male', 'female'], n_samples)
    
    # Test fairness metrics
    fairness = FairnessMetrics()
    eq_odds = fairness.equalized_odds(y_true, y_pred, sensitive_attr)
    dem_parity = fairness.demographic_parity(y_pred, sensitive_attr)
    
    print(f"\nFairness Metrics:")
    print(f"  Equalized Odds: {eq_odds:.4f}")
    print(f"  Demographic Parity: {dem_parity:.4f}")
    
    # Test explainability metrics
    explainability = ExplainabilityMetrics()
    shap_values = np.random.rand(11)
    shap_values[:2] = 0.0  # Suppress first two features
    
    sparsity = explainability.feature_sparsity(shap_values)
    print(f"\nExplainability Metrics:")
    print(f"  Feature Sparsity: {sparsity:.4f}")
    
    # Test aggregator
    aggregator = MetricsAggregator()
    thresholds = {
        'equalized_odds': 0.05,
        'demographic_parity': 0.05
    }
    
    metrics = {
        'equalized_odds': eq_odds,
        'demographic_parity': dem_parity
    }
    
    compliant, individual = aggregator.check_compliance(metrics, thresholds)
    print(f"\nCompliance Check:")
    print(f"  Overall: {'✓ Compliant' if compliant else '✗ Non-compliant'}")
    for metric, status in individual.items():
        print(f"  {metric}: {'✓' if status else '✗'}")