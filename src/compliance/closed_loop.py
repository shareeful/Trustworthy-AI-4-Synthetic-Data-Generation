"""
Closed-Loop Compliance Engine: Implements Algorithm 1.
Autonomously monitors metrics and adjusts loss weights during training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

from .metrics import MetricsAggregator

logger = logging.getLogger(__name__)


class ClosedLoopComplianceEngine:
    """
    Implements Algorithm 1: Closed-Loop Compliance Engine.
    
    Monitors T-AI metrics every N epochs and autonomously increments
    loss weights when violations are detected, ensuring convergence
    to regulatory requirements.
    """
    
    def __init__(
        self,
        profile_config: dict,
        initial_weights: Dict[str, float],
        monitor_frequency: int = 10,
        weight_increment: float = 0.1,
        max_cycles: int = 50,
        accuracy_floor: float = 0.60
    ):
        """
        Initialize Closed-Loop Engine.
        
        Args:
            profile_config: Profile configuration from profiles.yaml
            initial_weights: Initial loss weights from Translation Mechanism
            monitor_frequency: Epochs between compliance checks
            weight_increment: Adjustment step size (Δλ)
            max_cycles: Maximum adjustment cycles
            accuracy_floor: Minimum acceptable accuracy (safety guard)
        """
        self.profile_config = profile_config
        self.profile_name = profile_config['name']
        self.thresholds = profile_config['thresholds']
        
        self.current_weights = initial_weights.copy()
        self.initial_weights = initial_weights.copy()
        
        self.monitor_frequency = monitor_frequency
        self.weight_increment = weight_increment
        self.max_cycles = max_cycles
        self.accuracy_floor = accuracy_floor
        
        self.metrics_aggregator = MetricsAggregator()
        
        # History tracking for audit trail
        self.adjustment_history = []
        self.metrics_history = defaultdict(list)
        self.weights_history = defaultdict(list)
        
        self.cycle_count = 0
        self.violations_detected = []
        
        logger.info(f"Initialized Closed-Loop Engine for {self.profile_name}")
        logger.info(f"Monitor frequency: every {monitor_frequency} epochs")
        logger.info(f"Weight increment: Δλ = {weight_increment}")
        logger.info(f"Accuracy floor: {accuracy_floor:.1%}")
    
    def monitor_and_adjust(
        self,
        epoch: int,
        model: nn.Module,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        sensitive_attr: np.ndarray,
        device: torch.device,
        x_adv: Optional[torch.Tensor] = None,
        shap_values: Optional[np.ndarray] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Monitor compliance metrics and adjust weights if violations detected.
        
        This implements the core monitoring logic from Algorithm 1.
        
        Args:
            epoch: Current training epoch
            model: Current model state
            x_val: Validation data
            y_val: Validation labels
            sensitive_attr: Protected attribute values
            device: Computation device
            x_adv: Adversarial samples (for security monitoring)
            shap_values: SHAP values (for explainability monitoring)
            
        Returns:
            Tuple of (should_continue_training, current_weights)
        """
        # Only monitor at specified frequency
        if epoch % self.monitor_frequency != 0:
            return True, self.current_weights
        
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPLIANCE MONITORING - Epoch {epoch}")
        logger.info(f"{'='*80}")
        
        # Compute all metrics
        metrics = self.metrics_aggregator.compute_all_metrics(
            model=model,
            x_real=x_val,
            y_true=y_val,
            sensitive_attr=sensitive_attr,
            x_adv=x_adv,
            shap_values=shap_values,
            device=device
        )
        
        # Log current metrics
        logger.info("Current Metrics:")
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append(value)
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Check compliance
        all_compliant, individual_compliance = self.metrics_aggregator.check_compliance(
            metrics, self.thresholds
        )
        
        # Log compliance status
        logger.info("\nCompliance Status:")
        for metric, compliant in individual_compliance.items():
            status = "✓ PASS" if compliant else "✗ FAIL"
            threshold = self.thresholds.get(metric, "N/A")
            logger.info(f"  {metric}: {status} (threshold: {threshold})")
        
        # Safety guard: check accuracy floor
        if 'clean_accuracy' in metrics and metrics['clean_accuracy'] < self.accuracy_floor:
            logger.warning(f"⚠️  Accuracy {metrics['clean_accuracy']:.1%} below floor {self.accuracy_floor:.1%}")
            logger.warning("Stopping training to prevent utility collapse")
            return False, self.current_weights
        
        # If all compliant, we're done!
        if all_compliant:
            logger.info("\n🎉 ALL METRICS COMPLIANT - Training can stop")
            return False, self.current_weights
        
        # Otherwise, adjust weights for violated constraints
        violations = self._detect_violations(metrics, individual_compliance)
        
        if violations:
            self.cycle_count += 1
            logger.info(f"\n🔧 Adjustment Cycle {self.cycle_count}/{self.max_cycles}")
            logger.info(f"Violations detected: {violations}")
            
            # Store violation record
            self.violations_detected.append({
                'epoch': epoch,
                'cycle': self.cycle_count,
                'violations': violations,
                'metrics': metrics.copy()
            })
            
            # Adjust weights
            adjustments = self._adjust_weights(violations)
            
            # Log adjustments
            logger.info("\nWeight Adjustments:")
            for weight_name, (old_val, new_val) in adjustments.items():
                logger.info(f"  λ_{weight_name}: {old_val:.2f} → {new_val:.2f} (+{self.weight_increment})")
                self.weights_history[weight_name].append(new_val)
            
            # Store adjustment history for audit
            self.adjustment_history.append({
                'epoch': epoch,
                'cycle': self.cycle_count,
                'violations': violations,
                'adjustments': adjustments,
                'new_weights': self.current_weights.copy()
            })
            
            # Check if max cycles reached
            if self.cycle_count >= self.max_cycles:
                logger.warning(f"\n⚠️  Maximum cycles ({self.max_cycles}) reached")
                logger.warning("Stopping training - manual intervention may be required")
                return False, self.current_weights
        
        # Continue training with updated weights
        return True, self.current_weights
    
    def _detect_violations(
        self,
        metrics: Dict[str, float],
        compliance: Dict[str, bool]
    ) -> List[str]:
        """
        Detect which T-AI characteristics are violated.
        
        Args:
            metrics: Current metric values
            compliance: Compliance status for each metric
            
        Returns:
            List of violated characteristics: ['fairness', 'security', 'explainability']
        """
        violations = []
        
        # Map metrics to T-AI characteristics
        fairness_metrics = ['equalized_odds', 'demographic_parity']
        security_metrics = ['attack_success_rate']
        explainability_metrics = ['shap_rank_stability', 'feature_sparsity']
        
        # Check fairness violations
        if any(not compliance.get(m, True) for m in fairness_metrics if m in compliance):
            violations.append('fairness')
        
        # Check security violations
        if any(not compliance.get(m, True) for m in security_metrics if m in compliance):
            violations.append('security')
        
        # Check explainability violations
        if any(not compliance.get(m, True) for m in explainability_metrics if m in compliance):
            violations.append('explainability')
        
        return violations
    
    def _adjust_weights(self, violations: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Adjust loss weights for violated characteristics.
        
        Implements: λ_i ← λ_i + Δλ for violated characteristic i
        
        Args:
            violations: List of violated characteristics
            
        Returns:
            Dictionary of {weight_name: (old_value, new_value)}
        """
        adjustments = {}
        
        # Map violations to weight names
        violation_to_weight = {
            'fairness': 'fair',
            'security': 'adv',
            'explainability': 'sparse'
        }
        
        for violation in violations:
            weight_name = violation_to_weight.get(violation)
            if weight_name and weight_name in self.current_weights:
                old_value = self.current_weights[weight_name]
                new_value = old_value + self.weight_increment
                
                self.current_weights[weight_name] = new_value
                adjustments[weight_name] = (old_value, new_value)
        
        return adjustments
    
    def get_audit_certificate(self) -> Dict:
        """
        Generate compliance audit certificate with complete traceability.
        
        Returns:
            Dictionary containing full audit trail
        """
        certificate = {
            'profile': {
                'name': self.profile_name,
                'eu_article': self.profile_config['eu_article'],
                'target_characteristic': self.profile_config['target_characteristic']
            },
            'translation': {
                'initial_weights': self.initial_weights,
                'formula': f"λ = α·p + β (Translation Mechanism, Eq. 1)"
            },
            'closed_loop': {
                'total_cycles': self.cycle_count,
                'monitor_frequency': self.monitor_frequency,
                'weight_increment': self.weight_increment,
                'adjustment_history': self.adjustment_history,
                'violations_detected': self.violations_detected
            },
            'final_state': {
                'weights': self.current_weights,
                'weight_evolution': {
                    name: {
                        'initial': self.initial_weights.get(name, 0.0),
                        'final': self.current_weights.get(name, 0.0),
                        'delta': self.current_weights.get(name, 0.0) - self.initial_weights.get(name, 0.0)
                    }
                    for name in self.current_weights.keys()
                }
            },
            'compliance_evidence': {
                'thresholds': self.thresholds,
                'metrics_history': dict(self.metrics_history)
            }
        }
        
        return certificate
    
    def generate_audit_report(self) -> str:
        """
        Generate human-readable audit report.
        
        Returns:
            Formatted audit report string
        """
        cert = self.get_audit_certificate()
        
        report = f"""
{'='*80}
TRUSTWORTHY AI COMPLIANCE AUDIT CERTIFICATE
{'='*80}

PROFILE INFORMATION
-------------------
Name: {cert['profile']['name']}
EU AI Act Article: {cert['profile']['eu_article']}
Target Characteristic: {cert['profile']['target_characteristic']}

PHASE 3: TRANSLATION MECHANISM
-------------------------------
Initial Weights (Equation 1):
"""
        for weight_name, value in cert['translation']['initial_weights'].items():
            report += f"  λ_{weight_name} = {value:.2f}\n"
        
        report += f"\nPHASE 5: CLOSED-LOOP COMPLIANCE ENGINE\n"
        report += f"---------------------------------------\n"
        report += f"Total Adjustment Cycles: {cert['closed_loop']['total_cycles']}\n"
        report += f"Monitor Frequency: Every {cert['closed_loop']['monitor_frequency']} epochs\n"
        report += f"Weight Increment: Δλ = {cert['closed_loop']['weight_increment']}\n\n"
        
        if cert['closed_loop']['violations_detected']:
            report += "Violations Detected:\n"
            for i, violation in enumerate(cert['closed_loop']['violations_detected'], 1):
                report += f"  Cycle {i} (Epoch {violation['epoch']}): {violation['violations']}\n"
        else:
            report += "No violations detected - Initial configuration was compliant\n"
        
        report += f"\nFINAL STATE\n"
        report += f"-----------\n"
        report += f"Final Weights:\n"
        for weight_name, evolution in cert['final_state']['weight_evolution'].items():
            report += f"  λ_{weight_name}: {evolution['initial']:.2f} → {evolution['final']:.2f} "
            report += f"(Δ = {evolution['delta']:+.2f})\n"
        
        report += f"\nCOMPLIANCE EVIDENCE\n"
        report += f"-------------------\n"
        report += f"Thresholds:\n"
        for metric, threshold in cert['compliance_evidence']['thresholds'].items():
            report += f"  {metric}: {threshold}\n"
        
        report += f"\nFinal Metrics:\n"
        for metric_name, history in cert['compliance_evidence']['metrics_history'].items():
            if history:
                final_value = history[-1]
                report += f"  {metric_name}: {final_value:.4f}\n"
        
        report += f"\n{'='*80}\n"
        report += f"Certificate Generated: {self.profile_name}\n"
        report += f"Traceability: EU AI Act → Profile → Weights → Training → Compliance\n"
        report += f"{'='*80}\n"
        
        return report


if __name__ == "__main__":
    # Test Closed-Loop Engine
    import yaml
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    with open("config/profiles.yaml") as f:
        profiles = yaml.safe_load(f)
    
    # Initial weights (from Translation Mechanism)
    initial_weights = {
        'recon': 1.0,
        'fair': 3.5,
        'adv': 2.0,
        'sparse': 1.0
    }
    
    # Initialize engine
    engine = ClosedLoopComplianceEngine(
        profile_config=profiles['profile_p1'],
        initial_weights=initial_weights,
        monitor_frequency=10,
        weight_increment=0.1,
        max_cycles=50,
        accuracy_floor=0.60
    )
    
    print("\n" + "="*80)
    print("TESTING CLOSED-LOOP COMPLIANCE ENGINE")
    print("="*80)
    
    # Simulate monitoring cycles with violations
    print("\nSimulating training with compliance violations...\n")
    
    # Create dummy model and data for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(11, 11)
        
        def forward(self, x):
            return {'x_recon': torch.sigmoid(self.fc(x))}
    
    dummy_model = DummyModel()
    x_val = torch.randn(100, 11)
    y_val = torch.randint(0, 2, (100,)).float()
    sensitive_attr = np.random.choice(['male', 'female'], 100)
    device = torch.device('cpu')
    
    # Simulate 3 monitoring cycles
    for epoch in [10, 20, 30]:
        should_continue, weights = engine.monitor_and_adjust(
            epoch=epoch,
            model=dummy_model,
            x_val=x_val,
            y_val=y_val,
            sensitive_attr=sensitive_attr,
            device=device
        )
        
        print(f"\nEpoch {epoch} - Continue: {should_continue}, Weights: {weights}")
    
    # Generate audit certificate
    print("\n" + "="*80)
    print("AUDIT CERTIFICATE")
    print("="*80)
    print(engine.generate_audit_report())