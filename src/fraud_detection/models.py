import os
from typing import Any, Dict
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_logistic(X, y, random_state=42):
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X, y)
    return clf


def train_random_forest(X, y, random_state=42):
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X, y)
    return clf


def train_xgboost(X, y, random_state=42):
    try:
        import xgboost as xgb
    except Exception:
        raise
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    clf.fit(X, y)
    return clf


def evaluate_model(model: Any, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }


def save_model(model: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
