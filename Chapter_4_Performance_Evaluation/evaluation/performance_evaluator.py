# -*- coding: utf-8 -*-
"""Performance evaluator for OLE Protocol evaluation."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulators.performance_simulator import PerformanceSimulator, SimulationResult
from models.evaluation_models import NormalizationModel, ScoringModel, NormalizedMetrics, WeightSensitivityAnalyzer
from config.experiment_config import ExperimentConfig, WeightConfig


@dataclass
class EvaluationResult:
    """Container for complete evaluation results."""
    protocol_name: str
    # Raw metrics
    communication: float
    rounds: int
    computation: float
    security: float
    # Normalized metrics
    norm_communication: float
    norm_rounds: float
    norm_computation: float
    norm_security: float
    # Score decomposition
    comm_contribution: float
    round_contribution: float
    comp_contribution: float
    sec_contribution: float
    # Comprehensive score
    comprehensive_score: float
    # Rank
    rank: int = 0


class PerformanceEvaluator:
    """Coordinator for OLE protocol performance evaluation."""

    def __init__(self, random_seed: int = 42):
        self.simulator = PerformanceSimulator(random_seed)
        self.normalizer = NormalizationModel()
        self.config = ExperimentConfig()

    def evaluate_at_scale(self, scale: int, weight_config: WeightConfig = None,
                          n_trials: int = 100) -> Dict[str, EvaluationResult]:
        """Evaluate all protocols at a specific scale."""
        # Get weight configuration
        if weight_config is None:
            weight_config = self.config.weight_configs['Baseline']

        # Initialize scoring model
        scorer = ScoringModel(weight_config.to_array())

        # Simulate all protocols
        sim_results = {}
        for protocol in self.simulator.protocol_family.get_protocol_names():
            sim_results[protocol] = self.simulator.simulate_single_protocol(
                protocol, scale, n_trials
            )

        # Prepare metrics for normalization
        metrics_dict = {}
        for protocol, result in sim_results.items():
            metrics_dict[protocol] = {
                'communication': result.communication_mean,
                'rounds': result.total_rounds,
                'computation': result.computation_mean,
                'security': result.security_margin
            }

        # Normalize metrics
        normalized = self.normalizer.normalize_metrics(metrics_dict)

        # Compute scores and decomposition
        eval_results = {}
        for protocol in sim_results.keys():
            sim = sim_results[protocol]
            norm = normalized[protocol]
            decomp = scorer.compute_score_decomposition(norm)

            eval_results[protocol] = EvaluationResult(
                protocol_name=protocol,
                communication=sim.communication_mean,
                rounds=sim.total_rounds,
                computation=sim.computation_mean,
                security=sim.security_margin,
                norm_communication=norm.communication,
                norm_rounds=norm.rounds,
                norm_computation=norm.computation,
                norm_security=norm.security,
                comm_contribution=decomp['communication'],
                round_contribution=decomp['rounds'],
                comp_contribution=decomp['computation'],
                sec_contribution=decomp['security'],
                comprehensive_score=decomp['total']
            )

        # Assign ranks
        sorted_by_score = sorted(
            eval_results.items(),
            key=lambda x: x[1].comprehensive_score,
            reverse=True
        )
        for rank, (protocol, result) in enumerate(sorted_by_score, 1):
            eval_results[protocol].rank = rank

        return eval_results

    def generate_table12_data(self, scale: int = 16) -> Dict[str, dict]:
        """Generate normalized metrics data for Table 12."""
        results = self.evaluate_at_scale(scale)

        table_data = {}
        for protocol, result in results.items():
            table_data[protocol] = {
                'norm_comm': result.norm_communication,
                'norm_rounds': result.norm_rounds,
                'norm_comp': result.norm_computation,
                'norm_security': result.norm_security,
                'weighted_score': result.comprehensive_score
            }

        return table_data

    def generate_table13_data(self, scale: int = 16) -> Dict[str, Dict[str, float]]:
        """Generate comprehensive scores under different weight configs for Table 13."""
        # Simulate once
        sim_results = {}
        for protocol in self.simulator.protocol_family.get_protocol_names():
            sim_results[protocol] = self.simulator.simulate_single_protocol(
                protocol, scale, n_trials=100
            )

        # Prepare metrics
        metrics_dict = {}
        for protocol, result in sim_results.items():
            metrics_dict[protocol] = {
                'communication': result.communication_mean,
                'rounds': result.total_rounds,
                'computation': result.computation_mean,
                'security': result.security_margin
            }

        # Normalize
        normalized = self.normalizer.normalize_metrics(metrics_dict)

        # Compute scores for each weight config
        table_data = {}
        for protocol in sim_results.keys():
            table_data[protocol] = {}
            for config_name, weight_config in self.config.weight_configs.items():
                scorer = ScoringModel(weight_config.to_array())
                score = scorer.compute_comprehensive_score(normalized[protocol])
                table_data[protocol][config_name] = score

        return table_data

    def generate_table14_data(self, scale: int = 16) -> List[Dict]:
        """Generate complete ranking with score decomposition for Table 14."""
        results = self.evaluate_at_scale(scale)

        # Sort by rank
        sorted_results = sorted(results.values(), key=lambda x: x.rank)

        table_data = []
        for result in sorted_results:
            table_data.append({
                'rank': result.rank,
                'protocol': result.protocol_name,
                'overall_score': result.comprehensive_score,
                'comm_score': result.comm_contribution,
                'round_score': result.round_contribution,
                'comp_score': result.comp_contribution,
                'security_score': result.sec_contribution
            })

        return table_data

    def generate_radar_data(self, scale: int = 16) -> Dict[str, np.ndarray]:
        """Generate normalized metrics for radar chart (Figure 11)."""
        results = self.evaluate_at_scale(scale)

        radar_data = {}
        for protocol, result in results.items():
            radar_data[protocol] = np.array([
                result.norm_communication,
                result.norm_rounds,
                result.norm_computation,
                result.norm_security
            ])

        return radar_data

    def generate_sensitivity_data(self, scale: int = 16) -> Dict[str, np.ndarray]:
        """Generate weight sensitivity data for Figure 12."""
        table13 = self.generate_table13_data(scale)

        # Convert to arrays ordered by config
        config_order = ['Baseline', 'Comm.-Priority', 'Comp.-Priority', 'Balanced', 'Security-Priority']

        sensitivity_data = {}
        for protocol in table13.keys():
            scores = [table13[protocol][config] for config in config_order]
            sensitivity_data[protocol] = np.array(scores)

        return sensitivity_data

    def generate_decomposition_data(self, scale: int = 16) -> Dict[str, np.ndarray]:
        """Generate score decomposition data for Figure 13."""
        results = self.evaluate_at_scale(scale)

        # Sort by overall score (descending)
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].comprehensive_score,
            reverse=True
        )

        decomp_data = {}
        for protocol, result in sorted_results:
            decomp_data[protocol] = np.array([
                result.comm_contribution,
                result.round_contribution,
                result.comp_contribution,
                result.sec_contribution
            ])

        return decomp_data


class CorrelationAnalyzer:
    """Analyzer for framework validation through correlation analysis."""

    def __init__(self, evaluator: PerformanceEvaluator, simulator: PerformanceSimulator):
        self.evaluator = evaluator
        self.simulator = simulator

    def compute_prediction_correlation(self, scale: int = 16) -> Dict[str, float]:
        """Compute correlation between framework predictions and application performance."""
        from scipy import stats

        # Get framework scores
        eval_results = self.evaluator.evaluate_at_scale(scale)

        # Get application performance
        app_results = self.simulator.simulate_application_performance(scale)

        # Extract arrays
        protocols = list(eval_results.keys())
        scores = np.array([eval_results[p].comprehensive_score for p in protocols])
        times = np.array([app_results[p]['time_mean'] for p in protocols])
        time_stds = np.array([app_results[p]['time_std'] for p in protocols])

        # Compute correlation (scores should negatively correlate with time)
        correlation, p_value = stats.pearsonr(scores, times)

        # Compute linear regression for visualization
        slope, intercept, r_value, p_val, std_err = stats.linregress(scores, 1000/times)

        return {
            'protocols': protocols,
            'scores': scores,
            'times': times,
            'time_stds': time_stds,
            'correlation': r_value,
            'p_value': p_val,
            'slope': slope,
            'intercept': intercept,
            'std_err': std_err
        }

    def generate_table17_data(self, scale: int = 16) -> Dict[str, Dict]:
        """Generate application validation data for Table 17."""
        eval_results = self.evaluator.evaluate_at_scale(scale)
        app_results = self.simulator.simulate_application_performance(scale)

        # Determine observed ranks based on execution time
        protocols = list(app_results.keys())
        sorted_by_time = sorted(
            [(p, app_results[p]['time_mean']) for p in protocols],
            key=lambda x: x[1]
        )
        observed_ranks = {p: rank + 1 for rank, (p, _) in enumerate(sorted_by_time)}

        table_data = {}
        for protocol in protocols:
            predicted_rank = eval_results[protocol].rank
            obs_rank = observed_ranks[protocol]
            table_data[protocol] = {
                'framework_rank': predicted_rank,
                'observed_rank': obs_rank,
                'execution_time_mean': app_results[protocol]['time_mean'],
                'execution_time_std': app_results[protocol]['time_std'],
                'rank_match': 'Yes' if predicted_rank == obs_rank else 'Partial'
            }

        return table_data
