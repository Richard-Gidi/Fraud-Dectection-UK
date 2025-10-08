"""Feature engineering pipeline for financial behavioral data."""
import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.base import BaseEstimator, TransformerMixin


class FinancialFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate financial behavior features from transaction data."""
    
    def __init__(self, time_windows: List[str] = ['1h', '24h', '7d', '30d']):
        self.time_windows = time_windows
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
        features = []
        for col in ['amount', 'merchant_id', 'card_id']:
            if col in df.columns:
                # Rolling statistics
                for window in self.time_windows:
                    features.extend([
                        df.groupby('card_id')[col].rolling(window).mean(),
                        df.groupby('card_id')[col].rolling(window).std(),
                        df.groupby('card_id')[col].rolling(window).max(),
                        df.groupby('card_id')[col].rolling(window).count(),
                    ])
                    
        # Velocity checks
        for window in self.time_windows:
            features.extend([
                df.groupby('card_id').rolling(window).nunique()['merchant_id'],
                df.groupby('card_id').rolling(window)['amount'].agg(lambda x: (x > x.mean() + 2*x.std()).sum()),
            ])
            
        # Location/time features
        if 'merchant_zip' in df.columns:
            features.append(df.groupby('card_id')['merchant_zip'].rolling(window).nunique())
            
        df_features = pd.concat(features, axis=1)
        df_features.columns = [f'feature_{i}' for i in range(df_features.shape[1])]
        return df_features.fillna(0)


class AnomalyDetector:
    """Statistical anomaly detection for real-time fraud detection."""
    
    def __init__(self, contamination: float = 0.01):
        self.contamination = contamination
        self.threshold = None
        self.feature_generator = FinancialFeatureGenerator()
        
    def fit(self, X: pd.DataFrame):
        features = self.feature_generator.fit_transform(X)
        scores = features.sum(axis=1)  # Simple anomaly score
        self.threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        features = self.feature_generator.transform(X)
        scores = features.sum(axis=1)
        probs = 1 / (1 + np.exp(-(scores - self.threshold)))
        return np.vstack([1-probs, probs]).T