from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import uvicorn
import os

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent transactions using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class TransactionRequest(BaseModel):
    amount: float
    merchant_name: str
    merchant_category: str
    transaction_type: str
    card_present: bool
    timestamp: datetime
    distance_from_home: float
    distance_from_last_transaction: float
    ratio_to_median_purchase_price: float

# Load the model and preprocessor
try:
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'xgboost_model.joblib')
    preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'preprocessor.joblib')
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print(f"Model loaded from {model_path}")
    print(f"Preprocessor loaded from {preprocessor_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    preprocessor = None

@app.post("/api/v1/predict")
async def predict_fraud(transaction: TransactionRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame and prepare features
        data = pd.DataFrame([transaction.dict()])
        
        # Extract time-based features
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['month'] = data['timestamp'].dt.month
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # Map transaction type to TRANSACTION_TYPE_encoded
        data['TRANSACTION_TYPE_encoded'] = data['transaction_type'].map({
            'online': 1,
            'in_store': 0
        }).fillna(0)
        
        # Map card_present to TRANSACTION_DEVICE_encoded
        data['TRANSACTION_DEVICE_encoded'] = data['card_present'].astype(int)
        
        # Map amount to AMOUNT
        data['AMOUNT'] = data['amount']
        
        # Add default values for other required columns
        data['CHANNEL_encoded'] = 1  # Default channel (can be adjusted based on your encoding)
        data['TRANSACTION_STATUS_encoded'] = 1  # Default status for new transactions
        
        # Select and reorder columns for the preprocessor
        features = [
            'AMOUNT', 
            'month',
            'hour',
            'day_of_week',
            'TRANSACTION_TYPE_encoded',
            'TRANSACTION_DEVICE_encoded',
            'CHANNEL_encoded',
            'TRANSACTION_STATUS_encoded',
            'distance_from_home',
            'distance_from_last_transaction',
            'ratio_to_median_purchase_price'
        ]
        
        X = data[features]
        
        # Preprocess the data
        X_transformed = preprocessor.transform(X)
        
        # Make prediction
        prediction = model.predict(X_transformed)[0]
        probability = model.predict_proba(X_transformed)[0][1]
        
        return {
            "is_fraud": bool(prediction),
            "fraud_probability": float(probability),
            "model_version": "1.0.0",
            "features_used": features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Root endpoint that returns API status
    """
    return {"status": "online", "message": "Fraud Detection API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)