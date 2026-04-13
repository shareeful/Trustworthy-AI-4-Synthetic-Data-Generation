import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

from models.profiles import RegulatoryProfile


def baseline_preprocessing(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, num_cols


def compute_ipw_weights(df: pd.DataFrame, sensitive_col: str) -> np.ndarray:
    N = len(df)
    groups = df[sensitive_col].unique()
    C = len(groups)
    weights = np.ones(N)
    for g in groups:
        mask = df[sensitive_col] == g
        n_i = mask.sum()
        weights[mask] = N / (C * n_i)
    return weights


def remove_low_variance_features(df: pd.DataFrame,
                                  threshold: float = 0.01) -> pd.DataFrame:
    variances = df.var()
    keep = variances[variances >= threshold].index.tolist()
    return df[keep]


def preprocess(df: pd.DataFrame,
               profile: RegulatoryProfile,
               sensitive_col: str = "gender") -> Tuple[pd.DataFrame, np.ndarray]:
    df, _ = baseline_preprocessing(df)
    weights = np.ones(len(df))

    if "P-1" in profile.name:
        if sensitive_col in df.columns:
            weights = compute_ipw_weights(df, sensitive_col)

    if "P-3" in profile.name:
        df = remove_low_variance_features(df)

    return df, weights
