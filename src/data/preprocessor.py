"""
Profile-driven data preprocessing (Phase 2).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ProfileDrivenPreprocessor:
    """
    Implements Phase 2: Profile-Driven Data Preprocessing.
    Applies profile-specific transformations before model training.
    """
    
    def __init__(self, profile_config: dict):
        """
        Initialize preprocessor with profile configuration.
        
        Args:
            profile_config: Profile dictionary from profiles.yaml
        """
        self.profile_config = profile_config
        self.profile_name = profile_config['name']
        self.preprocessing_method = profile_config['preprocessing']['method']
        
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        self.ipw_weights = None
        
        logger.info(f"Initialized preprocessor for {self.profile_name}")
        logger.info(f"Preprocessing method: {self.preprocessing_method}")
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Fit preprocessor and transform data.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (features, target, sample_weights)
        """
        logger.info("Fitting preprocessor...")
        
        # Separate features and target
        target_col = 'charges'
        feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols].copy()
        y = df[target_col].values
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Standard preprocessing (all profiles)
        X = self._standard_preprocessing(X)
        
        # Profile-specific preprocessing
        sample_weights = None
        if self.preprocessing_method == 'inverse_probability_weighting':
            X, sample_weights = self._apply_ipw(X, df)
        elif self.preprocessing_method == 'feature_selection':
            X = self._apply_feature_selection(X, y)
        
        logger.info(f"Final feature shape: {X.shape}")
        
        return X, y, sample_weights
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform new data using fitted preprocessor."""
        target_col = 'charges'
        feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols].copy()
        y = df[target_col].values
        
        X = self._standard_preprocessing(X, fit=False)
        
        return X, y
    
    def _standard_preprocessing(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Apply standard preprocessing: encoding + normalization.
        
        Args:
            X: Feature dataframe
            fit: Whether to fit transformers
            
        Returns:
            Preprocessed feature array
        """
        # Identify column types
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        # Encode categorical variables
        X_encoded = X.copy()
        for col in categorical_cols:
            if fit:
                self.encoders[col] = LabelEncoder()
                X_encoded[col] = self.encoders[col].fit_transform(X[col])
            else:
                X_encoded[col] = self.encoders[col].transform(X[col])
        
        # Normalize numerical variables to [0, 1]
        if fit:
            self.scalers['features'] = MinMaxScaler()
            X_scaled = self.scalers['features'].fit_transform(X_encoded)
        else:
            X_scaled = self.scalers['features'].transform(X_encoded)
        
        logger.info(f"Standard preprocessing: {len(categorical_cols)} categorical, {len(numerical_cols)} numerical")
        
        return X_scaled
    
    def _apply_ipw(self, X: np.ndarray, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Inverse Probability Weighting for Profile P-1 (Fairness).
        
        Formula: w_i = N / (C * n_i)
        where N = total samples, C = number of groups, n_i = group size
        """
        logger.info("Applying Inverse Probability Weighting...")
        
        protected_attr = self.profile_config['preprocessing']['protected_attributes'][0]
        groups = df[protected_attr].values
        
        # Calculate IPW weights
        unique_groups = np.unique(groups)
        N = len(groups)
        C = len(unique_groups)
        
        weights = np.zeros(N)
        for group in unique_groups:
            group_mask = (groups == group)
            n_i = group_mask.sum()
            w_i = N / (C * n_i)
            weights[group_mask] = w_i
            logger.info(f"Group '{group}': n={n_i}, weight={w_i:.3f}")
        
        # Store for audit trail
        self.ipw_weights = {
            'male': weights[groups == 'male'][0] if 'male' in groups else None,
            'female': weights[groups == 'female'][0] if 'female' in groups else None
        }
        
        # Calculate demographic parity before/after
        dp_before = self._calculate_demographic_parity(df, weights=None)
        dp_after = self._calculate_demographic_parity(df, weights=weights)
        
        logger.info(f"Demographic Parity: {dp_before:.3f} → {dp_after:.3f} (Δ={dp_before-dp_after:.3f})")
        
        return X, weights
    
    def _apply_feature_selection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Apply feature selection for Profile P-3 (Explainability).
        Remove low-variance features to enhance interpretability.
        """
        logger.info("Applying feature selection...")
        
        from sklearn.ensemble import RandomForestRegressor
        
        # Train RF to get feature importances
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        threshold = self.profile_config['preprocessing']['importance_threshold']
        
        # Keep only important features
        important_features = importances >= threshold
        X_selected = X[:, important_features]
        
        logger.info(f"Feature selection: {X.shape[1]} → {X_selected.shape[1]} features")
        logger.info(f"Removed {(~important_features).sum()} low-importance features")
        
        return X_selected
    
    def _calculate_demographic_parity(self, df: pd.DataFrame, weights: Optional[np.ndarray] = None) -> float:
        """Calculate demographic parity metric."""
        # Simplified calculation for illustration
        # In real implementation, would use model predictions
        male_avg = df[df['gender'] == 'male']['charges'].mean()
        female_avg = df[df['gender'] == 'female']['charges'].mean()
        
        if weights is not None:
            male_mask = df['gender'] == 'male'
            female_mask = df['gender'] == 'female'
            male_avg = np.average(df[male_mask]['charges'], weights=weights[male_mask])
            female_avg = np.average(df[female_mask]['charges'], weights=weights[female_mask])
        
        dp = abs(male_avg - female_avg) / max(male_avg, female_avg)
        return dp
    
    def get_preprocessing_report(self) -> Dict:
        """Generate preprocessing report for audit trail."""
        report = {
            'profile': self.profile_name,
            'method': self.preprocessing_method,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
        }
        
        if self.ipw_weights:
            report['ipw_weights'] = self.ipw_weights
        
        return report


if __name__ == "__main__":
    # Test preprocessor
    import yaml
    logging.basicConfig(level=logging.INFO)
    
    # Load profile
    with open("config/profiles.yaml") as f:
        profiles = yaml.safe_load(f)
    
    # Load data
    from loader import InsuranceDataLoader
    loader = InsuranceDataLoader("data/insurance_claims.csv")
    train_df, _ = loader.load()
    train_df = loader.simulate_bias(train_df)
    
    # Test Profile P-1
    preprocessor = ProfileDrivenPreprocessor(profiles['profile_p1'])
    X, y, weights = preprocessor.fit_transform(train_df)
    
    print(f"\nPreprocessed data shape: {X.shape}")
    print(f"Sample weights shape: {weights.shape if weights is not None else 'None'}")
    print(f"\nPreprocessing report:\n{preprocessor.get_preprocessing_report()}")