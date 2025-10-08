import pytest
from src.fraud_detection import data, preprocess
import pandas as pd


def test_load_data_synthetic():
    df = data.load_data("nonexistent_file.csv")
    assert isinstance(df, pd.DataFrame)
    assert "label" in df.columns


def test_preprocessor_fit_transform():
    df = data.load_data("nonexistent.csv")
    X = df.drop(columns=["label"])[:100]
    numeric_features = list(X.columns)
    pre = preprocess.build_preprocessor(numeric_features)
    X_t, _ = preprocess.fit_transform_preprocessor(pre, X)
    assert X_t.shape[0] == X.shape[0]
