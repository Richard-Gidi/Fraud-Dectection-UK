"""Model interpretability using SHAP and LIME."""
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from .models import evaluate_model


class ModelInterpreter:
    """Wrapper for model interpretation using SHAP and LIME."""
    
    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
    def fit_shap_explainer(self, X_background: np.ndarray):
        """Initialize SHAP explainer with background dataset."""
        if hasattr(self.model, 'predict_proba'):
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                X_background
            )
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict, 
                X_background
            )
        return self
        
    def explain_prediction(self, X: np.ndarray) -> Dict[str, Any]:
        """Get SHAP and LIME explanations for a prediction."""
        if self.explainer is None:
            raise ValueError("Must call fit_shap_explainer first")
            
        # SHAP values
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
            
        # LIME explanation
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=self.feature_names,
            class_names=['legitimate', 'fraud'],
            mode='classification'
        )
        lime_exp = explainer.explain_instance(
            X[0], 
            self.model.predict_proba
        )
        
        return {
            'shap_values': shap_values,
            'feature_importance': dict(zip(
                self.feature_names,
                np.abs(shap_values).mean(0)
            )),
            'lime_explanation': lime_exp.as_list(),
        }
        
    def global_importance(self, X: np.ndarray) -> pd.DataFrame:
        """Get global feature importance using SHAP values."""
        if self.explainer is None:
            raise ValueError("Must call fit_shap_explainer first")
            
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values).mean(0)
        })
        return importance_df.sort_values('importance', ascending=False)