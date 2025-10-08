"""Stress testing and portfolio risk assessment."""
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed


class StressTester:
    """Portfolio stress testing and risk assessment."""
    
    def __init__(
        self, 
        model: BaseEstimator,
        n_scenarios: int = 1000,
        n_jobs: int = -1
    ):
        self.model = model
        self.n_scenarios = n_scenarios
        self.n_jobs = n_jobs
        
    def generate_stress_scenarios(
        self, 
        X: pd.DataFrame,
        features_to_stress: List[str],
        stress_factors: Dict[str, tuple] = None
    ) -> pd.DataFrame:
        """Generate stress test scenarios by perturbing features."""
        scenarios = []
        
        if stress_factors is None:
            stress_factors = {
                feat: (-0.5, 2.0) for feat in features_to_stress
            }
            
        for _ in range(self.n_scenarios):
            scenario = X.copy()
            for feat in features_to_stress:
                factor_range = stress_factors.get(feat, (-0.5, 2.0))
                factor = np.random.uniform(*factor_range)
                scenario[feat] *= (1 + factor)
            scenarios.append(scenario)
            
        return pd.concat(scenarios)
        
    def run_parallel_predictions(self, scenarios: pd.DataFrame) -> np.ndarray:
        """Run predictions in parallel for better performance."""
        def predict_chunk(chunk):
            return self.model.predict_proba(chunk)[:, 1]
            
        chunks = np.array_split(scenarios, self.n_jobs)
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_chunk)(chunk) for chunk in chunks
        )
        return np.concatenate(predictions)
        
    def assess_portfolio_risk(
        self,
        X: pd.DataFrame,
        features_to_stress: List[str],
        stress_factors: Dict[str, tuple] = None
    ) -> Dict[str, float]:
        """Run stress tests and compute risk metrics."""
        scenarios = self.generate_stress_scenarios(
            X, features_to_stress, stress_factors
        )
        
        probs = self.run_parallel_predictions(scenarios)
        
        # Reshape to (n_scenarios, n_samples)
        probs = probs.reshape(self.n_scenarios, len(X))
        
        return {
            'expected_loss_rate': float(probs.mean()),
            'var_95': float(np.percentile(probs.mean(axis=1), 95)),
            'var_99': float(np.percentile(probs.mean(axis=1), 99)),
            'max_portfolio_loss': float(probs.max()),
            'std_dev': float(probs.std()),
        }
        
    def identify_high_risk_segments(
        self,
        X: pd.DataFrame,
        segment_columns: List[str],
        features_to_stress: List[str]
    ) -> pd.DataFrame:
        """Identify high-risk segments under stress scenarios."""
        scenarios = self.generate_stress_scenarios(
            X, features_to_stress
        )
        
        probs = self.run_parallel_predictions(scenarios)
        probs = probs.reshape(self.n_scenarios, len(X))
        
        # Compute risk metrics per segment
        segment_risks = []
        for _, group in scenarios.groupby(segment_columns):
            idx = group.index
            group_probs = probs[:, idx]
            
            segment_risks.append({
                'segment': group[segment_columns].iloc[0].to_dict(),
                'mean_loss_rate': group_probs.mean(),
                'var_95': np.percentile(group_probs.mean(axis=1), 95),
                'size': len(idx),
            })
            
        return pd.DataFrame(segment_risks)