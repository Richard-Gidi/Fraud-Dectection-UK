# Fraud Detection — Production-ready README

This repository contains a production-ready scaffold for a fraud detection system. It includes data preprocessing, model training (Logistic, Random Forest, XGBoost), a FastAPI-based prediction service, notebooks for monitoring, and a CI/CD workflow for automated testing and Docker image publishing.

Table of contents
- Project layout
- Key files and responsibilities
- How the API prediction flow works
- Quick start: run locally (train, serve, test)
- Docker: build and run
- CI/CD: GitHub Actions
- Troubleshooting
- Next steps and recommended improvements

---

## Project layout (high level)

```
.
├─ app/                       # FastAPI application and utilities
│  ├─ main.py                 # Entrypoint + endpoints
│  ├─ api/                    # (optional) router modules
│  ├─ models/                 # pydantic schemas
│  └─ utils/                  # helpers (ml pipeline, config)
├─ models/                    # serialized model artifacts (joblib)
├─ src/fraud_detection/       # training/data/preprocessing code
├─ tests/                     # pytest test suite
├─ .github/workflows/ci-cd.yml # GitHub Actions CI/CD
├─ Dockerfile                 # production image definition
├─ requirements.txt           # pinned dependencies
└─ README.md                  # this document
```

## Key files and responsibilities

- `src/fraud_detection/train.py`: trains models, evaluates them, and saves the best model and preprocessor to `models/`.
- `src/fraud_detection/preprocess.py`: builds the preprocessing pipeline (scalers, encoders, feature selectors).
- `models/`: contains `preprocessor.joblib` and `xgboost_model.joblib` (or other trained model artifacts). The API loads these at startup.
- `app/main.py`: FastAPI app that:
  - loads model and preprocessor
  - exposes `POST /api/v1/predict` for predictions
  - adds derived features (hour/month/day_of_week) and encodes inputs to match preprocessor expectations
- `app/notebooks/model_evaluation.ipynb`: notebook for diagnostics (ROC, confusion matrix, SHAP, feature importance).
- `.github/workflows/ci-cd.yml`: runs tests, builds, and pushes a Docker image to Docker Hub on merges to `main` (requires secrets).

## How the API prediction flow works (high-level)

1. Client sends a JSON transaction to `POST /api/v1/predict`.
2. `app/main.py` performs light input validation via Pydantic and derives features such as `hour`, `month`, `day_of_week`.
3. It maps categorical values to the encoded numeric columns expected by the saved `preprocessor.joblib`.
4. The preprocessor transforms the prepared DataFrame into the numeric matrix expected by the model.
5. The model (XGBoost by default) returns `predict()` and `predict_proba()` and the API responds with a JSON containing `is_fraud`, `fraud_probability`, and `model_version`.

## Quick start: run locally (development) — Windows PowerShell

1) Create a virtual environment and activate it (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
# If you prefer installing minimal runtime:
pip install fastapi uvicorn[standard] pandas scikit-learn xgboost joblib python-dotenv
```

3) Train the models (this saves artifacts under `models/`):

```powershell
python -m src.fraud_detection.train --config .env
```

4) Start the API server (from repo root):

```powershell
$env:PYTHONPATH = '.'
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

5) Test the API (curl/Postman/Swagger UI):

- Swagger UI: http://127.0.0.1:8000/docs
- GET root: http://127.0.0.1:8000/
- POST predict: http://127.0.0.1:8000/api/v1/predict

Example payload (body):

```json
{
  "amount": 1000.0,
  "merchant_name": "Tech Store",
  "merchant_category": "Electronics",
  "transaction_type": "online",
  "card_present": false,
  "timestamp": "2025-09-28T10:00:00",
  "distance_from_home": 25.5,
  "distance_from_last_transaction": 30.0,
  "ratio_to_median_purchase_price": 2.1
}
```

If you get an error about missing columns during `preprocessor.transform()` then the preprocessor expects additional derived/encoded columns — update `app/main.py` to create these columns before calling the preprocessor. The README includes guidance for common mapped columns.

## Docker (production-like run)

1) Build image (from repo root):

```powershell
docker build -t fraud-detection-api .
```

2) Run container:

```powershell
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api
```

Notes:
- Ensure `models/` is present at build-time. Alternatively mount `models/` at runtime:

```powershell
docker run -d -p 8000:8000 -v ${PWD}:/app fraud-detection-api
```

## CI/CD (GitHub Actions)

The pipeline `.github/workflows/ci-cd.yml`:

- Runs on push & PR to `main`.
- Sets up Python and installs dependencies.
- Runs `pytest` and uploads coverage to Codecov.
- Builds and pushes Docker image to Docker Hub (requires secrets):

  - `DOCKER_USERNAME`
  - `DOCKER_PASSWORD`

Add those secrets to enable automatic Docker pushes.

## Troubleshooting (common issues)

- "Model not loaded": ensure `models/xgboost_model.joblib` and `models/preprocessor.joblib` exist. Run training if they're missing.
- "columns are missing" from the preprocessor: create the required derived/encoded columns in `app/main.py` before calling `preprocessor.transform()`.
- Connection refused / 127.0.0.1 refused: ensure uvicorn is running and bound to 127.0.0.1:8000. Check logs where you started the server for errors.
- Package/binary incompatibilities (NumPy vs compiled extensions): use virtualenv and pin `numpy<2` if necessary (some compiled wheels require numpy 1.x). Example: `pip install numpy==1.26.4`.

## Recommended next steps and improvements

- Add proper input validation to accept flexible payloads and return descriptive errors.
- Add model versioning metadata and an endpoint to list available model versions.
- Add authentication (API Key/JWT) and rate limiting for the production API.
- Add monitoring (Prometheus) and structured logging.
- Add unit/integration tests for the API using `pytest` and `httpx` test client.

---

If you'd like, I can:

- Generate a `tests/test_api.py` pytest file that uses FastAPI's test client to validate the `/api/v1/predict` path.
- Create a minimal `docker-compose.yml` for local development.
- Harden input mapping for more robust preprocessor compatibility.

Tell me which of the above you'd like me to implement next.
