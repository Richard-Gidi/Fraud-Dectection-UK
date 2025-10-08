"""Train models: logistic, random forest, xgboost"""
import argparse
from .config import get_config
from .data import load_data, train_test_split
from .preprocess import build_preprocessor, fit_transform_preprocessor
from .models import train_logistic, train_random_forest, train_xgboost, evaluate_model, save_model
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to .env file (optional)", default=None)
    args = parser.parse_args()

    cfg = get_config()
    print(f"Using data path: {cfg.data_path}")
    df = load_data(cfg.data_path)

    target = "label"
    if target not in df.columns:
        raise ValueError(f"Data must contain '{target}' column")

    X_train, X_test, y_train, y_test = train_test_split(df, target, cfg.test_size, cfg.random_seed)

    numeric_features = list(X_train.columns)
    preprocessor = build_preprocessor(numeric_features)
    X_train_t, X_test_t = fit_transform_preprocessor(preprocessor, X_train, X_test)

    results = {}

    # Logistic
    log = train_logistic(X_train_t, y_train, cfg.random_seed)
    results['logistic'] = evaluate_model(log, X_test_t, y_test)
    save_model(log, os.path.join(cfg.model_dir, 'logistic.joblib'))

    # Random Forest
    rf = train_random_forest(X_train_t, y_train, cfg.random_seed)
    results['random_forest'] = evaluate_model(rf, X_test_t, y_test)
    save_model(rf, os.path.join(cfg.model_dir, 'random_forest.joblib'))

    # XGBoost
    xgb = train_xgboost(X_train_t, y_train, cfg.random_seed)
    results['xgboost'] = evaluate_model(xgb, X_test_t, y_test)
    
    # Save models and preprocessor to the models directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the best model (XGBoost) and preprocessor
    save_model(xgb, os.path.join(models_dir, 'xgboost_model.joblib'))
    save_model(preprocessor, os.path.join(models_dir, 'preprocessor.joblib'))
    
    print("Results:")
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    main()
