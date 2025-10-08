import pandas as pd
import os
from typing import Tuple


def load_data(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    # fallback: generate synthetic dataset for demo
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y
    return df


def train_test_split(df: pd.DataFrame, target: str, test_size: float, random_state: int) -> Tuple:
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
