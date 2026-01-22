"""
Data loading utilities for insurance claims dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class InsuranceDataLoader:
    """Loads and validates the Insurance Claim Analysis dataset."""
    
    EXPECTED_COLUMNS = [
        'age', 'gender', 'bmi', 'children', 'smoker', 
        'region', 'charges'
    ]
    
    CATEGORICAL_COLUMNS = ['gender', 'smoker', 'region']
    NUMERICAL_COLUMNS = ['age', 'bmi', 'children']
    TARGET_COLUMN = 'charges'
    
    def __init__(self, filepath: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize data loader.
        
        Args:
            filepath: Path to CSV file
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        """
        self.filepath = Path(filepath)
        self.test_size = test_size
        self.random_state = random_state
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        logger.info(f"Initialized DataLoader for {filepath}")
    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load dataset and split into train/test.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Loading dataset...")
        df = pd.read_csv(self.filepath)
        
        # Validate columns
        missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
        # Basic statistics
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        
        # Handle missing values
        df = self._handle_missing(df)
        
        # Split train/test
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=df['gender']  # Stratify by protected attribute
        )
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        return train_df, test_df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataset."""
        # Numerical: median imputation
        for col in self.NUMERICAL_COLUMNS:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Imputed {col} with median: {median_val}")
        
        # Categorical: mode imputation
        for col in self.CATEGORICAL_COLUMNS:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"Imputed {col} with mode: {mode_val}")
        
        return df
    
    def get_feature_types(self) -> dict:
        """Return dictionary of feature types."""
        return {
            'numerical': self.NUMERICAL_COLUMNS,
            'categorical': self.CATEGORICAL_COLUMNS,
            'target': self.TARGET_COLUMN
        }
    
    def simulate_bias(self, df: pd.DataFrame, ratio: Tuple[float, float] = (0.65, 0.35)) -> pd.DataFrame:
        """
        Simulate demographic bias by skewing gender distribution.
        
        Args:
            df: Input dataframe
            ratio: Target (male, female) ratio
            
        Returns:
            Skewed dataframe
        """
        logger.info(f"Simulating bias with ratio {ratio}")
        
        male_df = df[df['gender'] == 'male']
        female_df = df[df['gender'] == 'female']
        
        total_samples = len(df)
        target_male = int(total_samples * ratio[0])
        target_female = int(total_samples * ratio[1])
        
        # Resample to achieve target ratio
        if len(male_df) < target_male:
            male_df = male_df.sample(n=target_male, replace=True, random_state=self.random_state)
        else:
            male_df = male_df.sample(n=target_male, replace=False, random_state=self.random_state)
        
        if len(female_df) < target_female:
            female_df = female_df.sample(n=target_female, replace=True, random_state=self.random_state)
        else:
            female_df = female_df.sample(n=target_female, replace=False, random_state=self.random_state)
        
        skewed_df = pd.concat([male_df, female_df]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        logger.info(f"Original distribution: Male={len(df[df['gender']=='male'])}, Female={len(df[df['gender']=='female'])}")
        logger.info(f"Skewed distribution: Male={len(skewed_df[skewed_df['gender']=='male'])}, Female={len(skewed_df[skewed_df['gender']=='female'])}")
        
        return skewed_df


if __name__ == "__main__":
    # Test data loader
    logging.basicConfig(level=logging.INFO)
    loader = InsuranceDataLoader("data/insurance_claims.csv")
    train_df, test_df = loader.load()
    
    # Simulate bias
    train_biased = loader.simulate_bias(train_df, ratio=(0.65, 0.35))
    print(f"\nBiased train set demographics:\n{train_biased['gender'].value_counts(normalize=True)}")