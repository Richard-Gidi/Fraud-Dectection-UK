"""FastAPI service for real-time fraud detection scoring."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from .features import FinancialFeatureGenerator
from .interpretability import ModelInterpreter
from .stress_testing import StressTester


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection scoring API with model interpretability",
    version="1.0.0"
)

# Load models and processors
MODEL_DIR = os.getenv('MODEL_DIR', './models')
feature_generator = FinancialFeatureGenerator()
models = {
    name: joblib.load(os.path.join(MODEL_DIR, f'{name}.joblib'))
    for name in ['logistic', 'random_forest', 'xgboost']
}


class Transaction(BaseModel):
    """Single transaction for scoring."""
    transaction_id: str
    timestamp: str
    amount: float
    merchant_id: str
    merchant_name: Optional[str]
    merchant_zip: Optional[str]
    card_id: str
    card_type: Optional[str]


class BatchTransactions(BaseModel):
    """Batch of transactions for scoring."""
    transactions: List[Transaction]


class StressTestConfig(BaseModel):
    """Configuration for stress testing."""
    features_to_stress: List[str]
    stress_factors: Optional[Dict[str, List[float]]]
    n_scenarios: Optional[int] = 1000


@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    """Score a single transaction for fraud probability."""
    # Convert to DataFrame
    df = pd.DataFrame([transaction.dict()])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Generate features
    features = feature_generator.transform(df)
    
    # Get predictions from all models
    predictions = {}
    explanations = {}
    
    for name, model in models.items():
        try:
            pred_proba = model.predict_proba(features)[:, 1][0]
            predictions[name] = float(pred_proba)
            
            # Get SHAP explanation for high-risk transactions
            if pred_proba > 0.7:  # High risk threshold
                interpreter = ModelInterpreter(model, list(features.columns))
                interpreter.fit_shap_explainer(features)
                explanations[name] = interpreter.explain_prediction(features.values)
                
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error predicting with {name}: {str(e)}"
            )
            
    return {
        "transaction_id": transaction.transaction_id,
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions,
        "explanations": explanations,
        "high_risk": any(p > 0.7 for p in predictions.values())
    }


@app.post("/batch-predict")
async def batch_predict(batch: BatchTransactions):
    """Score a batch of transactions."""
    if not batch.transactions:
        raise HTTPException(
            status_code=400,
            detail="No transactions provided"
        )
        
    # Convert to DataFrame
    df = pd.DataFrame([t.dict() for t in batch.transactions])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Generate features
    features = feature_generator.transform(df)
    
    # Get predictions from all models
    results = []
    for i, transaction in enumerate(batch.transactions):
        predictions = {
            name: float(model.predict_proba(features[i:i+1])[:, 1][0])
            for name, model in models.items()
        }
        
        results.append({
            "transaction_id": transaction.transaction_id,
            "predictions": predictions,
            "high_risk": any(p > 0.7 for p in predictions.values())
        })
        
    return {"results": results}


@app.post("/stress-test")
async def run_stress_test(config: StressTestConfig, batch: BatchTransactions):
    """Run stress tests on a batch of transactions."""
    # Convert to DataFrame
    df = pd.DataFrame([t.dict() for t in batch.transactions])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Generate features
    features = feature_generator.transform(df)
    
    stress_results = {}
    for name, model in models.items():
        tester = StressTester(
            model,
            n_scenarios=config.n_scenarios or 1000
        )
        
        stress_factors = (
            {k: tuple(v) for k, v in config.stress_factors.items()}
            if config.stress_factors else None
        )
        
        results = tester.assess_portfolio_risk(
            features,
            config.features_to_stress,
            stress_factors
        )
        
        stress_results[name] = results
        
    return {
        "stress_test_results": stress_results,
        "config": config.dict(),
        "timestamp": datetime.now().isoformat()
    }