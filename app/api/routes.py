from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import TransactionRequest, PredictionResponse
from app.utils.ml_model import ModelPipeline
from app.utils.config import Settings

router = APIRouter()
settings = Settings()
model_pipeline = ModelPipeline()

@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict if a transaction is fraudulent
    """
    try:
        prediction = model_pipeline.predict(transaction.dict())
        fraud_probability = model_pipeline.predict_proba(transaction.dict())
        
        return PredictionResponse(
            is_fraud=bool(prediction),
            fraud_probability=float(fraud_probability[0][1]),
            model_version=settings.model_version
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "model_version": settings.model_version}