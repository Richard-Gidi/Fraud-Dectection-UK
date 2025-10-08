from src.fraud_detection import models
from sklearn.datasets import make_classification


def make_small():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
    return X, y


def test_train_logistic():
    X, y = make_small()
    clf = models.train_logistic(X, y)
    assert hasattr(clf, "predict")


def test_train_random_forest():
    X, y = make_small()
    clf = models.train_random_forest(X, y)
    assert hasattr(clf, "predict")


def test_train_xgboost():
    try:
        import xgboost  # type: ignore
    except Exception:
        import pytest

        pytest.skip("xgboost not installed in this environment")

    X, y = make_small()
    clf = models.train_xgboost(X, y)
    assert hasattr(clf, "predict")
