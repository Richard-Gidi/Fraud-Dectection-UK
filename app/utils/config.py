from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    model_version: str = "1.0.0"
    model_path: str = "models/xgboost_model.joblib"
    preprocessor_path: str = "models/preprocessor.joblib"
    
    class Config:
        env_file = ".env"