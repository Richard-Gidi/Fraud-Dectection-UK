from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TransactionRequest(BaseModel):
    amount: float
    merchant_name: str
    merchant_category: str
    transaction_type: str
    card_present: bool
    timestamp: datetime
    distance_from_home: Optional[float] = None
    distance_from_last_transaction: Optional[float] = None
    ratio_to_median_purchase_price: Optional[float] = None

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    model_version: str

    class Config:
        schema_extra = {
            "example": {
                "is_fraud": False,
                "fraud_probability": 0.123,
                "model_version": "1.0.0"
            }
        }