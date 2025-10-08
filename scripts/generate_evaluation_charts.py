"""Generate evaluation charts for saved models and preprocessor.

Outputs (saved under reports/figures):
 - confusion_matrix.png
 - roc_curve.png
 - precision_recall.png
 - feature_importance_xgboost.png (if available)
 - feature_importance_rf.png (if available)
 - metrics_summary.txt

Run from repo root:
    python scripts/generate_evaluation_charts.py

The script expects:
 - models/preprocessor.joblib
 - models/xgboost_model.joblib or models/random_forest.joblib or models/logistic.joblib
 - data/test.csv (or tests/ provides a test set) — if missing, script will attempt to use a small synthetic sample.

"""
import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        pass

safe_print(f"Root: {ROOT}")

# Helper: load model and preprocessor
preprocessor_path = MODELS_DIR / "preprocessor.joblib"
model_paths = {
    "xgboost": MODELS_DIR / "xgboost_model.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
    "logistic": MODELS_DIR / "logistic.joblib",
}

if not preprocessor_path.exists():
    print(f"Preprocessor not found at {preprocessor_path}. Aborting. Put preprocessor.joblib into models/ and retry.")
    sys.exit(1)

try:
    preprocessor = joblib.load(preprocessor_path)
    safe_print("Loaded preprocessor")
except Exception as e:
    safe_print("Failed to load preprocessor:", e)
    raise

# pick best available model (xgboost > random_forest > logistic)
model = None
model_name = None
for name, path in model_paths.items():
    if path.exists():
        try:
            model = joblib.load(path)
            model_name = name
            safe_print(f"Loaded model {name} from {path}")
            break
        except Exception as e:
            safe_print(f"Failed to load model at {path}: {e}")
            continue

if model is None:
    print("No model artifacts found in models/. Place xgboost_model.joblib or random_forest.joblib or logistic.joblib and retry.")
    sys.exit(1)

# Load test data if available
test_csv = DATA_DIR / "test.csv"
if test_csv.exists():
    df = pd.read_csv(test_csv)
    print(f"Loaded test data from {test_csv} ({len(df)} rows)")
else:
    # Try tests/test_data.csv or tests/test.csv
    alt = ROOT / "tests" / "test_data.csv"
    if alt.exists():
        df = pd.read_csv(alt)
        print(f"Loaded test data from {alt} ({len(df)} rows)")
    else:
        print("No test.csv found. Creating a small synthetic sample to demonstrate charts.")
        # create small sample with required columns
        df = pd.DataFrame([
            {
                "amount": 100.0,
                "merchant_name": "A",
                "merchant_category": "cat",
                "transaction_type": "online",
                "card_present": False,
                "timestamp": "2025-09-28T10:00:00",
                "distance_from_home": 5.0,
                "distance_from_last_transaction": 10.0,
                "ratio_to_median_purchase_price": 1.2,
                "is_fraud": 0,
            },
            {
                "amount": 5000.0,
                "merchant_name": "B",
                "merchant_category": "lux",
                "transaction_type": "online",
                "card_present": False,
                "timestamp": "2025-09-28T02:00:00",
                "distance_from_home": 200.0,
                "distance_from_last_transaction": 300.0,
                "ratio_to_median_purchase_price": 10.0,
                "is_fraud": 1,
            },
        ])

# Prepare features same as app/main.py
# ensure timestamp and derived features
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
else:
    df["timestamp"] = pd.to_datetime(pd.Timestamp.now())

# derive
if "hour" not in df.columns:
    df["hour"] = df["timestamp"].dt.hour
if "month" not in df.columns:
    df["month"] = df["timestamp"].dt.month
if "day_of_week" not in df.columns:
    df["day_of_week"] = df["timestamp"].dt.dayofweek

# encodings
if "TRANSACTION_TYPE_encoded" not in df.columns:
    df["TRANSACTION_TYPE_encoded"] = df.get("transaction_type", pd.Series(["online"]*len(df))).map({"online":1, "in_store":0}).fillna(0).astype(int)
if "TRANSACTION_DEVICE_encoded" not in df.columns:
    if "card_present" in df.columns:
        df["TRANSACTION_DEVICE_encoded"] = df["card_present"].astype(int)
    else:
        df["TRANSACTION_DEVICE_encoded"] = 0

if "AMOUNT" not in df.columns:
    df["AMOUNT"] = df.get("amount", 0.0)

if "CHANNEL_encoded" not in df.columns:
    df["CHANNEL_encoded"] = 1
if "TRANSACTION_STATUS_encoded" not in df.columns:
    df["TRANSACTION_STATUS_encoded"] = 1

features = [
    "AMOUNT",
    "month",
    "hour",
    "day_of_week",
    "TRANSACTION_TYPE_encoded",
    "TRANSACTION_DEVICE_encoded",
    "CHANNEL_encoded",
    "TRANSACTION_STATUS_encoded",
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
]

# Check columns exist, fill missing numeric with 0
for c in features:
    if c not in df.columns:
        df[c] = 0

X = df[features]

# transform
try:
    X_t = preprocessor.transform(X)
except Exception as e:
    print(f"Error during preprocessing.transform: {e}")
    # try to convert X to numeric
    X = X.fillna(0)
    X_t = preprocessor.transform(X)

# y true
if "is_fraud" in df.columns:
    y_true = df["is_fraud"].astype(int).values
else:
    # if we don't have labels, we cannot compute confusion/roc; create dummy
    y_true = np.zeros(len(df), dtype=int)

# predict
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_t)[:, 1]
else:
    # fallback: decision_function or predict
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_t)
        # convert scores to 0-1 via logistic
        y_proba = 1 / (1 + np.exp(-scores))
    else:
        y_proba = model.predict(X_t)
        # ensure float
        y_proba = np.array(y_proba, dtype=float)

# predicted labels using 0.5 threshold (note: user should pick threshold)
y_pred = (y_proba >= 0.5).astype(int)

# Metrics summary
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
print("Classification report:")
print(report)

with open(ROOT / "reports" / "metrics_summary.txt", "w") as f:
    f.write(str(report))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(4,4))
disp.plot(ax=ax)
ax.set_title(f"Confusion matrix ({model_name})")
fig.savefig(OUT_DIR / "confusion_matrix.png", bbox_inches='tight')
plt.close(fig)
print("Saved confusion matrix")

# ROC curve
if len(np.unique(y_true)) > 1:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0,1],[0,1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    fig.savefig(OUT_DIR / "roc_curve.png", bbox_inches='tight')
    plt.close(fig)
    print("Saved ROC curve")
else:
    print("ROC curve skipped (only one class in y_true)")

# Precision-Recall
if len(np.unique(y_true)) > 1:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    fig.savefig(OUT_DIR / "precision_recall.png", bbox_inches='tight')
    plt.close(fig)
    print("Saved precision-recall curve")
else:
    print("Precision-recall skipped (only one class in y_true)")

# Feature importance for tree models
try:
    if model_name == 'xgboost':
        # XGBoost scikit-learn wrapper exposes feature_importances_
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # try to get booster
            try:
                booster = model.get_booster()
                fmap = booster.get_score(importance_type='weight')
                # map feature names
                importances = np.zeros(X_t.shape[1])
                # can't map easily without feature names; skip
            except Exception:
                importances = None
        if importances is not None:
            # try to map to feature names from preprocessor if available
            try:
                # if preprocessor is a ColumnTransformer with named transformers
                feature_names = []
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out(features)
                else:
                    # fallback to raw features
                    feature_names = features
                # select top features
                idx = np.argsort(importances)[::-1][:20]
                fig, ax = plt.subplots(figsize=(8,6))
                ax.barh([feature_names[i] for i in idx[::-1]], importances[idx[::-1]])
                ax.set_title('XGBoost feature importances')
                fig.savefig(OUT_DIR / 'feature_importance_xgboost.png', bbox_inches='tight')
                plt.close(fig)
                print('Saved XGBoost feature importances')
    elif model_name == 'random_forest':
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            try:
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out(features)
                else:
                    feature_names = features
            except Exception:
                feature_names = features
            idx = np.argsort(importances)[::-1][:20]
            fig, ax = plt.subplots(figsize=(8,6))
            ax.barh([feature_names[i] for i in idx[::-1]], importances[idx[::-1]])
            ax.set_title('Random Forest feature importances')
            fig.savefig(OUT_DIR / 'feature_importance_rf.png', bbox_inches='tight')
            plt.close(fig)
            print('Saved Random Forest feature importances')
except Exception as e:
    print('Feature importance plotting failed:', e)

print('Done. Charts saved to', OUT_DIR)
