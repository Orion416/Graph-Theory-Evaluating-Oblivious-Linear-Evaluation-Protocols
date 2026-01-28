# -*- coding: utf-8 -*-
"""Comparator for analyzing model predictions vs real measurements."""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ComparisonResult:
    """Result of comparison between model and real data."""
    pearson_correlation: float          # Pearson r
    pearson_pvalue: float               # Pearson p-value
    spearman_correlation: float         # Spearman rho
    spearman_pvalue: float              # Spearman p-value
    kendall_tau: float                  # Kendall tau
    kendall_pvalue: float               # Kendall p-value
    mean_absolute_error: float          # MAE
    mean_absolute_percentage_error: float  # MAPE
    root_mean_squared_error: float      # RMSE
    ranking_match: bool                 # Exact ranking match
    ranking_top3_match: bool            # Top-3 ranking match
    per_protocol_errors: Dict[str, float] = field(default_factory=dict)

    def is_valid_model(self, threshold: float = 0.7) -> bool:
        """Check if model is valid based on correlation threshold."""
        return (abs(self.pearson_correlation) >= threshold and
                abs(self.spearman_correlation) >= threshold)

    def get_summary(self) -> str:
        """Generate summary report."""
        return f"""
        Model Validation Summary:
        ----------------
        Pearson r = {self.pearson_correlation:.4f} (p = {self.pearson_pvalue:.4e})
        Spearman rho = {self.spearman_correlation:.4f} (p = {self.spearman_pvalue:.4e})
        Kendall tau = {self.kendall_tau:.4f} (p = {self.kendall_pvalue:.4e})
        MAPE = {self.mean_absolute_percentage_error:.2%}
        RMSE = {self.root_mean_squared_error:.4f}
        Exact Match: {'Yes' if self.ranking_match else 'No'}
        Top-3 Match: {'Yes' if self.ranking_top3_match else 'No'}
        """


class ModelRealComparator:
    """Comparator for model predictions vs real measurements."""

    def __init__(self,
                 model_predictions: Dict[str, Dict[str, float]],
                 real_measurements: Dict[str, Dict[str, float]]):
        """Initialize comparator with predictions and measurements."""
        self.model = model_predictions
        self.real = real_measurements

        # Get common protocols
        self.protocols = list(
            set(model_predictions.keys()) & set(real_measurements.keys())
        )
        self.protocols.sort()

        if len(self.protocols) < 2:
            raise ValueError("Need at least 2 common protocols for comparison")

    def compare_communication(self) -> ComparisonResult:
        """Compare communication predictions."""
        return self._compare_metric('communication_kb')

    def compare_computation(self) -> ComparisonResult:
        """Compare computation predictions."""
        return self._compare_metric('computation_time_ms')

    def compare_total_time(self) -> ComparisonResult:
        """Compare total time predictions."""
        return self._compare_metric('total_time_ms')

    def compare_ranking(self, metric: str = 'total_score') -> ComparisonResult:
        """Compare ranking consistency."""
        model_values = np.array([self.model[p].get(metric, 0) for p in self.protocols])
        real_values = np.array([self.real[p].get(metric, 0) for p in self.protocols])

        # Calculate ranks (higher value = better rank, use negative for ascending sort)
        model_ranks = stats.rankdata([-v for v in model_values])
        real_ranks = stats.rankdata([-v for v in real_values])

        # Calculate correlations
        tau, tau_pvalue = stats.kendalltau(model_ranks, real_ranks)
        spearman, spearman_pvalue = stats.spearmanr(model_ranks, real_ranks)

        # Check match
        ranking_match = np.array_equal(model_ranks, real_ranks)
        top3_match = np.array_equal(
            np.argsort(model_ranks)[:3],
            np.argsort(real_ranks)[:3]
        )

        return ComparisonResult(
            pearson_correlation=np.nan,
            pearson_pvalue=np.nan,
            spearman_correlation=spearman,
            spearman_pvalue=spearman_pvalue,
            kendall_tau=tau,
            kendall_pvalue=tau_pvalue,
            mean_absolute_error=np.mean(np.abs(model_ranks - real_ranks)),
            mean_absolute_percentage_error=np.nan,
            root_mean_squared_error=np.sqrt(np.mean((model_ranks - real_ranks) ** 2)),
            ranking_match=ranking_match,
            ranking_top3_match=top3_match,
            per_protocol_errors={
                p: abs(mr - rr) for p, mr, rr in zip(self.protocols, model_ranks, real_ranks)
            }
        )

    def _compare_metric(self, metric: str) -> ComparisonResult:
        """Compare single metric."""
        model_values = np.array([self.model[p].get(metric, 0) for p in self.protocols])
        real_values = np.array([self.real[p].get(metric, 0) for p in self.protocols])

        # Filter valid data
        valid_mask = (real_values > 0) & (model_values > 0)
        if not np.any(valid_mask):
            raise ValueError(f"No valid data for metric {metric}")

        model_valid = model_values[valid_mask]
        real_valid = real_values[valid_mask]
        valid_protocols = [p for p, v in zip(self.protocols, valid_mask) if v]

        # Calculate correlations
        if len(model_valid) >= 2:
            pearson, pearson_p = stats.pearsonr(model_valid, real_valid)
            spearman, spearman_p = stats.spearmanr(model_valid, real_valid)
            tau, tau_p = stats.kendalltau(model_valid, real_valid)
        else:
            pearson = spearman = tau = np.nan
            pearson_p = spearman_p = tau_p = 1.0

        # Calculate errors
        errors = model_valid - real_valid
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        # Calculate percentage errors
        ape_list = []
        per_protocol_errors = {}
        for p, pred, actual in zip(valid_protocols, model_valid, real_valid):
            ape = abs(pred - actual) / actual if actual != 0 else float('inf')
            ape_list.append(ape)
            per_protocol_errors[p] = ape

        mape = np.mean(ape_list) if ape_list else float('inf')

        # Calculate ranking match
        model_ranks = stats.rankdata([-v for v in model_valid])
        real_ranks = stats.rankdata([-v for v in real_valid])
        ranking_match = np.array_equal(model_ranks, real_ranks)
        top3_match = np.array_equal(
            np.argsort(model_ranks)[:min(3, len(model_ranks))],
            np.argsort(real_ranks)[:min(3, len(real_ranks))]
        )

        return ComparisonResult(
            pearson_correlation=pearson,
            pearson_pvalue=pearson_p,
            spearman_correlation=spearman,
            spearman_pvalue=spearman_p,
            kendall_tau=tau,
            kendall_pvalue=tau_p,
            mean_absolute_error=mae,
            mean_absolute_percentage_error=mape,
            root_mean_squared_error=rmse,
            ranking_match=ranking_match,
            ranking_top3_match=top3_match,
            per_protocol_errors=per_protocol_errors
        )

    def compute_calibration_factors(self) -> Dict[str, Dict[str, float]]:
        """Compute model calibration factors."""
        calibration = {}
        metrics = ['communication_kb', 'computation_time_ms']

        for protocol in self.protocols:
            calibration[protocol] = {}
            for metric in metrics:
                model_val = self.model[protocol].get(metric, 0)
                real_val = self.real[protocol].get(metric, 0)

                if model_val > 0 and real_val > 0:
                    calibration[protocol][metric] = real_val / model_val
                else:
                    calibration[protocol][metric] = 1.0

        return calibration

    def generate_report(self) -> str:
        """Generate complete comparison report."""
        comm_result = self.compare_communication()
        comp_result = self.compare_computation()
        rank_result = self.compare_ranking()

        report = f"""
# Model vs Real Measurement Comparison Report

## 1. Communication Accuracy

| Metric | Value |
|---|---|
| Pearson r | {comm_result.pearson_correlation:.4f} |
| Spearman rho | {comm_result.spearman_correlation:.4f} |
| Kendall tau | {comm_result.kendall_tau:.4f} |
| MAPE | {comm_result.mean_absolute_percentage_error:.2%} |
| RMSE | {comm_result.root_mean_squared_error:.4f} |

## 2. Computation Accuracy

| Metric | Value |
|---|---|
| Pearson r | {comp_result.pearson_correlation:.4f} |
| Spearman rho | {comp_result.spearman_correlation:.4f} |
| Kendall tau | {comp_result.kendall_tau:.4f} |
| MAPE | {comp_result.mean_absolute_percentage_error:.2%} |
| RMSE | {comp_result.root_mean_squared_error:.4f} |

## 3. Ranking Consistency

| Metric | Value |
|---|---|
| Kendall tau | {rank_result.kendall_tau:.4f} |
| Spearman rho | {rank_result.spearman_correlation:.4f} |
| Exact Match | {'Yes' if rank_result.ranking_match else 'No'} |
| Top-3 Match | {'Yes' if rank_result.ranking_top3_match else 'No'} |

## 4. Per-Protocol Errors

| Protocol | Comm Error | Comp Error |
|---|---|---|
"""
        for p in self.protocols:
            comm_err = comm_result.per_protocol_errors.get(p, 'N/A')
            comp_err = comp_result.per_protocol_errors.get(p, 'N/A')
            if isinstance(comm_err, float):
                comm_err = f"{comm_err:.2%}"
            if isinstance(comp_err, float):
                comp_err = f"{comp_err:.2%}"
            report += f"| {p} | {comm_err} | {comp_err} |\n"

        report += f"""

## 5. Model Validity

- Communication Model: {'Valid' if comm_result.is_valid_model() else 'Needs Improvement'}
- Computation Model: {'Valid' if comp_result.is_valid_model() else 'Needs Improvement'}
- Ranking Capability: {'Reliable' if rank_result.kendall_tau > 0.6 else 'Needs Verification'}

## 6. Recommendations

"""
        if not comm_result.is_valid_model():
            report += "- Calibrate communication model with real data\n"
        if not comp_result.is_valid_model():
            report += "- Introduce real parameters for computation model\n"
        if rank_result.kendall_tau < 0.6:
            report += "- Ranking unstable, increase sample size\n"

        return report

    def export_comparison_data(self) -> Dict:
        """Export comparison data for visualization."""
        return {
            'protocols': self.protocols,
            'model_predictions': self.model,
            'real_measurements': self.real,
            'communication_comparison': self.compare_communication().__dict__,
            'computation_comparison': self.compare_computation().__dict__,
            'ranking_comparison': self.compare_ranking().__dict__,
            'calibration_factors': self.compute_calibration_factors()
        }
