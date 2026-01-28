# -*- coding: utf-8 -*-
"""Model vs Measurement Comparison Module"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy import stats


@dataclass
class ComparisonResult:
    """Comparison result for a single metric (model vs. real)"""
    metric_name: str

    # Correlation metrics
    pearson_correlation: float      # Pearson correlation coefficient
    spearman_correlation: float     # Spearman rank correlation coefficient
    kendall_tau: float              # Kendall's Tau

    # Error metrics
    mean_absolute_error: float      # MAE
    mean_absolute_percentage_error: float  # MAPE
    root_mean_square_error: float   # RMSE

    # Ranking consistency
    ranking_match: bool             # Whether the ranking is exactly the same
    rank_differences: Dict[str, int]  # Ranking differences for each protocol

    # Per-protocol errors
    per_protocol_errors: Dict[str, float]


@dataclass
class ValidationReport:
    """Complete validation report"""
    communication_comparison: ComparisonResult
    computation_comparison: ComparisonResult
    ranking_comparison: ComparisonResult

    overall_validity: bool          # Overall validity judgment
    validity_reasons: List[str]     # Reasons for validity judgment

    @property
    def summary_score(self) -> float:
        """Calculate comprehensive validation score"""
        comm_score = max(0, self.communication_comparison.spearman_correlation)
        comp_score = max(0, self.computation_comparison.spearman_correlation)
        rank_score = max(0, self.ranking_comparison.kendall_tau)
        return (comm_score + comp_score + rank_score) / 3


class ModelRealComparator:
    """Comparator for model predictions vs. real measurements"""

    def __init__(
        self,
        model_predictions: Dict[str, Dict[str, float]],
        real_measurements: Dict[str, Dict[str, float]]
    ):
        """Initialize the comparator"""
        self.model = model_predictions
        self.real = real_measurements

        # Get common list of protocols
        self.protocols = sorted(
            set(model_predictions.keys()) & set(real_measurements.keys())
        )

    def _compute_correlations(
        self,
        model_values: np.ndarray,
        real_values: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute three types of correlation coefficients"""
        if len(model_values) < 2:
            return 0.0, 0.0, 0.0

        # Handle possible constant arrays
        if np.std(model_values) == 0 or np.std(real_values) == 0:
            return 0.0, 0.0, 0.0

        pearson, _ = stats.pearsonr(model_values, real_values)
        spearman, _ = stats.spearmanr(model_values, real_values)
        kendall, _ = stats.kendalltau(model_values, real_values)

        return pearson, spearman, kendall

    def _compute_errors(
        self,
        model_values: np.ndarray,
        real_values: np.ndarray,
        protocols: List[str]
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """Compute error metrics"""
        mae = np.mean(np.abs(model_values - real_values))

        # MAPE - avoid division by zero
        ape_list = []
        per_protocol_errors = {}

        for i, pid in enumerate(protocols):
            if real_values[i] != 0:
                ape = abs(model_values[i] - real_values[i]) / real_values[i]
            else:
                ape = 0.0 if model_values[i] == 0 else float('inf')

            ape_list.append(ape)
            per_protocol_errors[pid] = ape

        mape = np.mean([e for e in ape_list if e != float('inf')])
        rmse = np.sqrt(np.mean((model_values - real_values) ** 2))

        return mae, mape, rmse, per_protocol_errors

    def _compute_rank_differences(
        self,
        model_values: np.ndarray,
        real_values: np.ndarray,
        protocols: List[str]
    ) -> Tuple[bool, Dict[str, int]]:
        """Compute ranking differences"""
        # Rank in descending order (larger value means higher rank)
        model_ranks = stats.rankdata(-model_values, method='ordinal')
        real_ranks = stats.rankdata(-real_values, method='ordinal')

        rank_diffs = {}
        for i, pid in enumerate(protocols):
            rank_diffs[pid] = int(abs(model_ranks[i] - real_ranks[i]))

        ranking_match = all(d == 0 for d in rank_diffs.values())

        return ranking_match, rank_diffs

    def compare_metric(self, metric_name: str) -> ComparisonResult:
        """Compare a single metric"""
        model_values = []
        real_values = []

        for pid in self.protocols:
            model_val = self.model[pid].get(metric_name, 0)
            real_val = self.real[pid].get(metric_name, 0)
            model_values.append(model_val)
            real_values.append(real_val)

        model_values = np.array(model_values)
        real_values = np.array(real_values)

        # Compute correlations
        pearson, spearman, kendall = self._compute_correlations(
            model_values, real_values
        )

        # Compute errors
        mae, mape, rmse, per_protocol_errors = self._compute_errors(
            model_values, real_values, self.protocols
        )

        # Compute ranking differences
        ranking_match, rank_diffs = self._compute_rank_differences(
            model_values, real_values, self.protocols
        )

        return ComparisonResult(
            metric_name=metric_name,
            pearson_correlation=pearson,
            spearman_correlation=spearman,
            kendall_tau=kendall,
            mean_absolute_error=mae,
            mean_absolute_percentage_error=mape,
            root_mean_square_error=rmse,
            ranking_match=ranking_match,
            rank_differences=rank_diffs,
            per_protocol_errors=per_protocol_errors
        )

    def compare_ranking(self, score_metric: str = 'total_score') -> ComparisonResult:
        """Compare protocol ranking consistency"""
        return self.compare_metric(score_metric)

    def generate_validation_report(self) -> ValidationReport:
        """Generate a complete validation report"""
        # Compare communication
        comm_result = self.compare_metric('communication')

        # Compare computation
        comp_result = self.compare_metric('computation')

        # Compare ranking
        rank_result = self.compare_ranking('total_score')

        # Judgment of overall validity
        validity_reasons = []

        # Validity criteria
        comm_valid = comm_result.spearman_correlation > 0.6
        comp_valid = comp_result.spearman_correlation > 0.5
        rank_valid = rank_result.kendall_tau > 0.5

        if comm_valid:
            validity_reasons.append(
                f"Communication prediction Spearman correlation ({comm_result.spearman_correlation:.2f}) > 0.6"
            )
        else:
            validity_reasons.append(
                f"Communication prediction Spearman correlation ({comm_result.spearman_correlation:.2f}) <= 0.6 (Failed)"
            )

        if comp_valid:
            validity_reasons.append(
                f"Computation prediction Spearman correlation ({comp_result.spearman_correlation:.2f}) > 0.5"
            )
        else:
            validity_reasons.append(
                f"Computation prediction Spearman correlation ({comp_result.spearman_correlation:.2f}) <= 0.5 (Failed)"
            )

        if rank_valid:
            validity_reasons.append(
                f"Ranking consistency Kendall's Tau ({rank_result.kendall_tau:.2f}) > 0.5"
            )
        else:
            validity_reasons.append(
                f"Ranking consistency Kendall's Tau ({rank_result.kendall_tau:.2f}) <= 0.5 (Failed)"
            )

        # Overall valid: at least two criteria met
        overall_valid = sum([comm_valid, comp_valid, rank_valid]) >= 2

        return ValidationReport(
            communication_comparison=comm_result,
            computation_comparison=comp_result,
            ranking_comparison=rank_result,
            overall_validity=overall_valid,
            validity_reasons=validity_reasons
        )

    def print_report(self, report: ValidationReport) -> str:
        """Generate printable report text"""
        lines = []
        lines.append("=" * 80)
        lines.append("Model Validation Report")
        lines.append("=" * 80)

        lines.append("\n1. Communication Prediction Accuracy")
        lines.append("-" * 60)
        cr = report.communication_comparison
        lines.append(f"   Pearson Correlation:  {cr.pearson_correlation:.4f}")
        lines.append(f"   Spearman Correlation: {cr.spearman_correlation:.4f}")
        lines.append(f"   Kendall's Tau:        {cr.kendall_tau:.4f}")
        lines.append(f"   MAPE:                 {cr.mean_absolute_percentage_error:.2%}")

        lines.append("\n2. Computation Prediction Accuracy")
        lines.append("-" * 60)
        cr = report.computation_comparison
        lines.append(f"   Pearson Correlation:  {cr.pearson_correlation:.4f}")
        lines.append(f"   Spearman Correlation: {cr.spearman_correlation:.4f}")
        lines.append(f"   Kendall's Tau:        {cr.kendall_tau:.4f}")
        lines.append(f"   MAPE:                 {cr.mean_absolute_percentage_error:.2%}")

        lines.append("\n3. Protocol Ranking Consistency")
        lines.append("-" * 60)
        cr = report.ranking_comparison
        lines.append(f"   Kendall's Tau:        {cr.kendall_tau:.4f}")
        lines.append(f"   Perfect Match:        {'Yes' if cr.ranking_match else 'No'}")

        lines.append("\n4. Validation Conclusion")
        lines.append("-" * 60)
        lines.append(f"   Summary Score:    {report.summary_score:.2f}")
        lines.append(f"   Overall Validity: {'Passed' if report.overall_validity else 'Failed'}")

        lines.append("\n   Criteria:")
        for reason in report.validity_reasons:
            lines.append(f"   - {reason}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


def create_comparison_from_graphs_and_benchmarks(
    graphs: dict,
    metrics: dict,
    benchmark_results: dict
) -> ModelRealComparator:
    """Create comparator from graph models and benchmark results"""
    model_predictions = {}
    real_measurements = {}

    for pid in graphs.keys():
        graph = graphs[pid]
        metric = metrics.get(pid)

        if not metric:
            continue

        # Model predicted values
        model_predictions[pid] = {
            'communication': graph.total_weight / 8,  # bits -> bytes
            'computation': metric.avg_edge_weight * metric.num_edges / 1000,  # Simplified computation metric
            'total_score': graph.total_weight,  # Used for ranking
        }

        # Real measured values (take average of first instance quantity)
        if pid in benchmark_results:
            instance_results = list(benchmark_results[pid].values())
            if instance_results:
                measurements = instance_results[0]  # Take first instance quantity
                avg_comm = np.mean([m.communication_bytes for m in measurements])
                avg_comp = np.mean([m.computation_time_ms for m in measurements])

                real_measurements[pid] = {
                    'communication': avg_comm,
                    'computation': avg_comp,
                    'total_score': avg_comm + avg_comp * 100,  # Comprehensive score
                }

    return ModelRealComparator(model_predictions, real_measurements)


if __name__ == "__main__":
    # Simple test
    model = {
        'P1': {'communication': 1000, 'computation': 50, 'total_score': 1050},
        'P2': {'communication': 1500, 'computation': 30, 'total_score': 1530},
        'P3': {'communication': 500, 'computation': 80, 'total_score': 580},
    }

    real = {
        'P1': {'communication': 1100, 'computation': 55, 'total_score': 1155},
        'P2': {'communication': 1400, 'computation': 35, 'total_score': 1435},
        'P3': {'communication': 550, 'computation': 75, 'total_score': 625},
    }

    comparator = ModelRealComparator(model, real)
    report = comparator.generate_validation_report()
    print(comparator.print_report(report))
