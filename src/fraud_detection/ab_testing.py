"""A/B testing and model performance validation."""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

logger = logging.getLogger(__name__)


class ABTester:
    """A/B testing for model performance validation."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.results = {}
        
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray = None
    ) -> Dict[str, float]:
        """Compute key classification metrics."""
        return {
            'auc_roc': float(roc_auc_score(y_true, y_pred, sample_weight=sample_weight)),
            'auc_pr': float(average_precision_score(y_true, y_pred, sample_weight=sample_weight)),
            'capture_rate': float((y_pred > 0.5).mean()),
            'precision': float(np.sum((y_pred > 0.5) & (y_true == 1)) / max(1, np.sum(y_pred > 0.5))),
        }
        
    def compare_models(
        self,
        model_a_preds: np.ndarray,
        model_b_preds: np.ndarray,
        y_true: np.ndarray,
        segment: str = 'all'
    ) -> Dict[str, Dict]:
        """Compare two models using bootstrap hypothesis testing."""
        metrics_a = self._compute_metrics(y_true, model_a_preds)
        metrics_b = self._compute_metrics(y_true, model_b_preds)
        
        # Bootstrap comparison
        n_bootstrap = 1000
        n_samples = len(y_true)
        
        bootstrap_diff = {metric: [] for metric in metrics_a.keys()}
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            metrics_a_boot = self._compute_metrics(
                y_true[idx], model_a_preds[idx]
            )
            metrics_b_boot = self._compute_metrics(
                y_true[idx], model_b_preds[idx]
            )
            
            for metric in metrics_a.keys():
                bootstrap_diff[metric].append(
                    metrics_a_boot[metric] - metrics_b_boot[metric]
                )
                
        # Compute p-values and confidence intervals
        results = {
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
            'differences': {},
            'significant': {},
            'confidence_intervals': {}
        }
        
        for metric in metrics_a.keys():
            diff_dist = np.array(bootstrap_diff[metric])
            
            # Two-sided p-value
            p_value = min(
                2 * np.mean(diff_dist <= 0),
                2 * np.mean(diff_dist >= 0)
            )
            
            results['differences'][metric] = float(
                metrics_a[metric] - metrics_b[metric]
            )
            results['significant'][metric] = p_value < self.significance_level
            results['confidence_intervals'][metric] = [
                float(np.percentile(diff_dist, 2.5)),
                float(np.percentile(diff_dist, 97.5))
            ]
            
        self.results[segment] = results
        return results
        
    def compare_segments(
        self,
        model_preds: np.ndarray,
        y_true: np.ndarray,
        segments: pd.Series
    ) -> pd.DataFrame:
        """Compare model performance across different segments."""
        segment_results = []
        
        for segment in segments.unique():
            mask = segments == segment
            if mask.sum() < 100:  # Skip small segments
                continue
                
            metrics = self._compute_metrics(
                y_true[mask],
                model_preds[mask]
            )
            
            segment_results.append({
                'segment': segment,
                'size': int(mask.sum()),
                **metrics
            })
            
        return pd.DataFrame(segment_results)
        
    def get_summary(self, detailed: bool = False) -> str:
        """Generate a summary of A/B test results."""
        summary = []
        
        for segment, result in self.results.items():
            summary.append(f"\nResults for segment: {segment}")
            
            for metric in result['metrics_a'].keys():
                diff = result['differences'][metric]
                ci = result['confidence_intervals'][metric]
                significant = result['significant'][metric]
                
                summary.append(
                    f"\n{metric}:"
                    f"\n  Model A: {result['metrics_a'][metric]:.4f}"
                    f"\n  Model B: {result['metrics_b'][metric]:.4f}"
                    f"\n  Difference: {diff:.4f} ({ci[0]:.4f}, {ci[1]:.4f})"
                    f"\n  Significant: {significant}"
                )
                
            if not detailed:
                break
                
        return "\n".join(summary)