from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing import Tuple, List


def build_preprocessor(features: List[str]) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric features."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("features", pipeline, features),
    ], remainder='drop')


def fit_transform_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame, X_val: pd.DataFrame = None) -> Tuple:
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = None
    if X_val is not None:
        X_val_t = preprocessor.transform(X_val)
    return X_train_t, X_val_t
