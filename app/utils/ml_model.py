import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
from app.utils.config import Settings

class ModelPipeline:
    def __init__(self):
        self.settings = Settings()
        self.model_path = Path(self.settings.model_path)
        self.preprocessor_path = Path(self.settings.preprocessor_path)
        
        # Load model and preprocessor
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        
    def _preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess a single transaction
        """
        df = pd.DataFrame([data])
        
        # Extract time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Drop timestamp after feature extraction
        df = df.drop('timestamp', axis=1)
        
        # Transform using preprocessor
        return pd.DataFrame(
            self.preprocessor.transform(df),
            columns=self.preprocessor.get_feature_names_out()
        )
    
    def predict(self, data: Dict[str, Any]) -> int:
        """
        Make a prediction for a single transaction
        """
        X = self._preprocess_input(data)
        return self.model.predict(X)[0]
    
    def predict_proba(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Get prediction probabilities for a single transaction
        """
        X = self._preprocess_input(data)
        return self.model.predict_proba(X)