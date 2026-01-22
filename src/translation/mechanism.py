"""
Translation Mechanism: Derives loss weights from regulatory profiles (Equation 1).
This implements Phase 3 of the methodology.
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TranslationMechanism:
    """
    Implements the Translation Mechanism from Equation 1:
    
    [λ_Recon]       [1.0                    ]
    [λ_Fair  ]  =   [α · p_Fair + β         ]
    [λ_Adv   ]       [α · p_Sec + β          ]
    [λ_Sparse]       [α · p_Expl + β         ]
    
    where α = 5.0 (scaling factor), β = 0.5 (bias)
    """
    
    def __init__(self, alpha: float = 5.0, beta: float = 0.5):
        """
        Initialize Translation Mechanism.
        
        Args:
            alpha: Scaling factor for T-AI priorities (default: 5.0)
            beta: Bias to prevent complete constraint neglect (default: 0.5)
        """
        self.alpha = alpha
        self.beta = beta
        
        logger.info(f"Initialized Translation Mechanism: α={alpha}, β={beta}")
    
    def translate(self, profile_config: dict) -> Dict[str, float]:
        """
        Translate regulatory profile to loss function weights.
        
        Args:
            profile_config: Profile dictionary from profiles.yaml
            
        Returns:
            Dictionary of loss weights: {recon, fair, adv, sparse}
        """
        priorities = profile_config['priorities']
        p_fair = priorities['fairness']
        p_sec = priorities['security']
        p_expl = priorities['explainability']
        
        # Validate priorities sum to 1.0
        total = p_fair + p_sec + p_expl
        if not np.isclose(total, 1.0):
            logger.warning(f"Priorities sum to {total}, expected 1.0. Normalizing...")
            p_fair /= total
            p_sec /= total
            p_expl /= total
        
        # Apply Equation 1
        weights = {
            'recon': 1.0,  # Baseline reconstruction fidelity
            'fair': self.alpha * p_fair + self.beta,
            'adv': self.alpha * p_sec + self.beta,
            'sparse': self.alpha * p_expl + self.beta
        }
        
        # Log translation
        logger.info(f"Profile: {profile_config['name']}")
        logger.info(f"Priorities: Fair={p_fair:.1f}, Sec={p_sec:.1f}, Expl={p_expl:.1f}")
        logger.info(f"Derived weights: λ_Recon={weights['recon']:.1f}, λ_Fair={weights['fair']:.1f}, "
                   f"λ_Adv={weights['adv']:.1f}, λ_Sparse={weights['sparse']:.1f}")
        
        # Store for audit trail
        self._last_translation = {
            'profile': profile_config['name'],
            'eu_article': profile_config['eu_article'],
            'priorities': priorities,
            'weights': weights,
            'formula': f"λ = α·p + β, where α={self.alpha}, β={self.beta}"
        }
        
        return weights
    
    def get_audit_trail(self) -> Dict:
        """
        Get audit trail of last translation.
        
        Returns:
            Dictionary with profile, priorities, weights, and formula
        """
        if not hasattr(self, '_last_translation'):
            raise ValueError("No translation performed yet")
        
        return self._last_translation
    
    def validate_weights(self, weights: Dict[str, float]) -> bool:
        """
        Validate that weights are within reasonable ranges.
        
        Args:
            weights: Dictionary of weights
            
        Returns:
            True if valid, False otherwise
        """
        # Check all weights are positive
        for key, val in weights.items():
            if val < 0:
                logger.error(f"Invalid weight {key}={val} (must be positive)")
                return False
        
        # Check reconstruction weight is baseline
        if weights['recon'] != 1.0:
            logger.warning(f"Reconstruction weight is {weights['recon']}, expected 1.0")
        
        # Check T-AI weights are in reasonable range [0.5, 5.5]
        for key in ['fair', 'adv', 'sparse']:
            if not (0.5 <= weights[key] <= 5.5):
                logger.warning(f"Weight {key}={weights[key]:.2f} outside expected range [0.5, 5.5]")
        
        return True
    
    def explain_weight(self, weight_name: str, weight_value: float, priority: float) -> str:
        """
        Generate human-readable explanation for a weight.
        
        Args:
            weight_name: Name of the weight (e.g., 'fair')
            weight_value: Computed weight value
            priority: Priority from profile (0-1)
            
        Returns:
            Explanation string
        """
        explanations = {
            'fair': 'Fairness enforcement (non-discrimination)',
            'adv': 'Security enforcement (adversarial robustness)',
            'sparse': 'Explainability enforcement (feature sparsity)'
        }
        
        description = explanations.get(weight_name, weight_name)
        
        explanation = (
            f"{description}:\n"
            f"  Priority: {priority:.1%} → Weight: {weight_value:.2f}\n"
            f"  Calculation: λ = {self.alpha} × {priority:.2f} + {self.beta} = {weight_value:.2f}\n"
            f"  Impact: {description} violations are penalized {weight_value:.1f}× "
            f"more than reconstruction errors"
        )
        
        return explanation
    
    def compare_profiles(self, profiles: list) -> str:
        """
        Generate comparison table of multiple profiles.
        
        Args:
            profiles: List of profile configurations
            
        Returns:
            Formatted comparison string
        """
        comparison = "Profile Comparison:\n"
        comparison += "=" * 80 + "\n"
        comparison += f"{'Profile':<20} {'Fair λ':<10} {'Sec λ':<10} {'Expl λ':<10} {'Target':<20}\n"
        comparison += "-" * 80 + "\n"
        
        for profile in profiles:
            weights = self.translate(profile)
            comparison += (
                f"{profile['name']:<20} "
                f"{weights['fair']:<10.2f} "
                f"{weights['adv']:<10.2f} "
                f"{weights['sparse']:<10.2f} "
                f"{profile['target_characteristic']:<20}\n"
            )
        
        comparison += "=" * 80 + "\n"
        
        return comparison


if __name__ == "__main__":
    # Test Translation Mechanism
    import yaml
    logging.basicConfig(level=logging.INFO)
    
    # Load profiles
    with open("config/profiles.yaml") as f:
        profiles_config = yaml.safe_load(f)
    
    # Initialize mechanism
    translator = TranslationMechanism(alpha=5.0, beta=0.5)
    
    # Test Profile P-1 (Fairness)
    print("\n" + "="*80)
    print("TESTING PROFILE P-1 (HIGH-RISK MEDICAL)")
    print("="*80)
    weights_p1 = translator.translate(profiles_config['profile_p1'])
    print(f"\nDerived weights: {weights_p1}")
    
    # Get audit trail
    audit = translator.get_audit_trail()
    print(f"\nAudit Trail:")
    print(f"  EU Article: {audit['eu_article']}")
    print(f"  Formula: {audit['formula']}")
    
    # Explain fairness weight
    print("\n" + translator.explain_weight('fair', weights_p1['fair'], 0.6))
    
    # Test Profile P-2 (Security)
    print("\n" + "="*80)
    print("TESTING PROFILE P-2 (SAFETY-CRITICAL)")
    print("="*80)
    weights_p2 = translator.translate(profiles_config['profile_p2'])
    print(f"\nDerived weights: {weights_p2}")
    
    # Test Profile P-3 (Explainability)
    print("\n" + "="*80)
    print("TESTING PROFILE P-3 (AUDIT-READY)")
    print("="*80)
    weights_p3 = translator.translate(profiles_config['profile_p3'])
    print(f"\nDerived weights: {weights_p3}")
    
    # Compare all profiles
    print("\n")
    all_profiles = [
        profiles_config['profile_p1'],
        profiles_config['profile_p2'],
        profiles_config['profile_p3']
    ]
    print(translator.compare_profiles(all_profiles))
    
    # Validate weights
    print("\nValidation:")
    for profile_name, profile in profiles_config.items():
        weights = translator.translate(profile)
        is_valid = translator.validate_weights(weights)
        print(f"  {profile['name']}: {'✓ Valid' if is_valid else '✗ Invalid'}")