"""Model governance and compliance reporting framework."""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path
import yaml

from .interpretability import ModelInterpreter
from .ab_testing import ABTester


class ModelGovernance:
    """Model governance and compliance reporting framework."""
    
    def __init__(
        self,
        model_name: str,
        model_version: str,
        governance_path: str = "governance"
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.governance_path = Path(governance_path)
        self.governance_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        
    def log_training_metadata(
        self,
        params: Dict,
        metrics: Dict[str, float],
        feature_importance: Dict[str, float]
    ):
        """Log model training metadata for compliance."""
        metadata = {
            "model_name": self.model_name,
            "version": self.model_version,
            "timestamp": datetime.now().isoformat(),
            "parameters": params,
            "metrics": metrics,
            "feature_importance": feature_importance
        }
        
        path = self.governance_path / f"{self.model_name}_training_{self.model_version}.json"
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Logged training metadata to {path}")
        
    def log_prediction(
        self,
        transaction_id: str,
        features: Dict,
        prediction: float,
        explanation: Optional[Dict] = None
    ):
        """Log individual predictions for audit."""
        log_entry = {
            "model_name": self.model_name,
            "version": self.model_version,
            "timestamp": datetime.now().isoformat(),
            "transaction_id": transaction_id,
            "features": features,
            "prediction": float(prediction)
        }
        
        if explanation:
            log_entry["explanation"] = explanation
            
        path = self.governance_path / f"predictions_{self.model_version}.jsonl"
        with open(path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def generate_model_card(
        self,
        description: str,
        intended_use: List[str],
        limitations: List[str],
        ethical_considerations: List[str],
        training_data: Dict,
        evaluation_data: Dict,
        quantitative_analyses: Dict,
    ):
        """Generate a model card for documentation."""
        model_card = {
            "model_details": {
                "name": self.model_name,
                "version": self.model_version,
                "date": datetime.now().isoformat(),
                "description": description,
            },
            "intended_use": {
                "primary_uses": intended_use,
                "out_of_scope": limitations
            },
            "factors": {
                "relevant_factors": ethical_considerations,
            },
            "metrics": {
                "performance_measures": quantitative_analyses
            },
            "training_data": training_data,
            "evaluation_data": evaluation_data,
        }
        
        path = self.governance_path / f"{self.model_name}_card_{self.model_version}.yaml"
        with open(path, 'w') as f:
            yaml.dump(model_card, f, default_flow_style=False)
            
        return model_card
        
    def generate_compliance_report(
        self,
        start_date: str,
        end_date: str
    ) -> Dict:
        """Generate compliance report for a time period."""
        # Load prediction logs
        predictions_file = self.governance_path / f"predictions_{self.model_version}.jsonl"
        
        if not predictions_file.exists():
            return {"error": "No prediction logs found"}
            
        predictions = []
        with open(predictions_file) as f:
            for line in f:
                pred = json.loads(line)
                if start_date <= pred['timestamp'] <= end_date:
                    predictions.append(pred)
                    
        if not predictions:
            return {"error": "No predictions in specified date range"}
            
        # Analyze predictions
        df = pd.DataFrame(predictions)
        
        report = {
            "model_info": {
                "name": self.model_name,
                "version": self.model_version,
                "report_period": {
                    "start": start_date,
                    "end": end_date
                }
            },
            "summary_statistics": {
                "total_predictions": len(df),
                "average_score": float(df['prediction'].mean()),
                "high_risk_rate": float((df['prediction'] > 0.7).mean())
            },
            "performance_monitoring": {
                "score_distribution": df['prediction'].describe().to_dict()
            }
        }
        
        # Add feature importance if available
        if 'explanation' in df.columns:
            importance = pd.DataFrame([
                x['feature_importance'] 
                for x in df['explanation'] 
                if x and 'feature_importance' in x
            ]).mean()
            
            report["feature_importance"] = importance.to_dict()
            
        return report
        
    def validate_model_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        threshold: float = 0.1
    ) -> Dict:
        """Check for model and data drift."""
        from scipy import stats
        
        drift_metrics = {}
        
        # Feature drift
        for col in reference_data.columns:
            try:
                # Kolmogorov-Smirnov test for distribution shift
                ks_stat, p_value = stats.ks_2samp(
                    reference_data[col],
                    current_data[col]
                )
                
                drift_metrics[f"feature_drift_{col}"] = {
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_value),
                    "is_drift": p_value < 0.05
                }
            except Exception as e:
                self.logger.warning(f"Could not compute drift for {col}: {e}")
                
        # Population Stability Index (PSI)
        def compute_psi(expected, actual, bins=10):
            cuts = np.percentile(
                np.concatenate([expected, actual]),
                np.linspace(0, 100, bins + 1)
            )
            
            expected_hist = np.histogram(expected, bins=cuts)[0] / len(expected)
            actual_hist = np.histogram(actual, bins=cuts)[0] / len(actual)
            
            # Add small epsilon to prevent division by zero
            expected_hist = np.clip(expected_hist, 1e-10, None)
            actual_hist = np.clip(actual_hist, 1e-10, None)
            
            return float(np.sum(
                (actual_hist - expected_hist) * np.log(actual_hist / expected_hist)
            ))
            
        drift_metrics["population_stability"] = {
            "psi": compute_psi(
                reference_data.mean(axis=1),
                current_data.mean(axis=1)
            ),
            "is_significant_drift": False  # Set based on threshold
        }
        
        return drift_metrics