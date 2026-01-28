# -*- coding: utf-8 -*-
"""Evaluation models for OLE Protocol Performance Evaluation."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import load_yaml_config, ExperimentConfig


@dataclass
class NormalizedMetrics:
    """Container for normalized performance metrics."""
    communication: float
    rounds: float
    computation: float
    security: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.communication,
            self.rounds,
            self.computation,
            self.security
        ])


class NormalizationModel:
    """Normalization model for multi-dimensional performance metrics."""

    def __init__(self):
        self.metrics_cache = {}

    def normalize_communication(self, values: np.ndarray) -> np.ndarray:
        """Normalize communication values (lower is better)."""
        log_values = np.log10(values + 1)
        min_val = np.min(log_values)
        max_val = np.max(log_values)

        if max_val - min_val < 1e-10:
            return np.ones_like(values)

        normalized = 1 - (log_values - min_val) / (max_val - min_val)
        return normalized

    def normalize_rounds(self, values: np.ndarray) -> np.ndarray:
        """Normalize round values (lower is better)."""
        min_val = np.min(values)
        max_val = np.max(values)

        if max_val - min_val < 1e-10:
            return np.ones_like(values)

        normalized = 1 - (values - min_val) / (max_val - min_val)
        return normalized

    def normalize_computation(self, values: np.ndarray) -> np.ndarray:
        """Normalize computation values (lower is better)."""
        log_values = np.log10(values + 1)
        min_val = np.min(log_values)
        max_val = np.max(log_values)

        if max_val - min_val < 1e-10:
            return np.ones_like(values)

        normalized = 1 - (log_values - min_val) / (max_val - min_val)
        return normalized

    def normalize_security(self, values: np.ndarray) -> np.ndarray:
        """Normalize security margin values (higher is better)."""
        min_val = np.min(values)
        max_val = np.max(values)

        if max_val - min_val < 1e-10:
            return np.ones_like(values)

        normalized = (values - min_val) / (max_val - min_val)
        return normalized

    def normalize_metrics(self, metrics_dict: Dict[str, dict]) -> Dict[str, NormalizedMetrics]:
        """Normalize metrics across all protocols."""
        protocols = list(metrics_dict.keys())
        n_protocols = len(protocols)

        comm_values = np.array([metrics_dict[p]['communication'] for p in protocols])
        round_values = np.array([metrics_dict[p]['rounds'] for p in protocols])
        comp_values = np.array([metrics_dict[p]['computation'] for p in protocols])
        sec_values = np.array([metrics_dict[p]['security'] for p in protocols])

        norm_comm = self.normalize_communication(comm_values)
        norm_rounds = self.normalize_rounds(round_values)
        norm_comp = self.normalize_computation(comp_values)
        norm_sec = self.normalize_security(sec_values)

        results = {}
        for i, protocol in enumerate(protocols):
            results[protocol] = NormalizedMetrics(
                communication=norm_comm[i],
                rounds=norm_rounds[i],
                computation=norm_comp[i],
                security=norm_sec[i]
            )

        return results


class ScoringModel:
    """Scoring model for comprehensive performance evaluation."""

    def __init__(self, weights: np.ndarray = None):
        """Initialize scoring model."""
        if weights is None:
            config = ExperimentConfig()
            baseline = config.weight_configs.get('Baseline')
            if baseline:
                self.weights = baseline.to_array()
                self._weight_source = "config/parameters.yaml (Baseline)"
            else:
                self.weights = np.array([0.35, 0.20, 0.25, 0.20])
                self._weight_source = "Fallback defaults"
        else:
            self.weights = weights
            self._weight_source = "User specified"

        if not np.isclose(np.sum(self.weights), 1.0):
            self.weights = self.weights / np.sum(self.weights)

    def get_weight_source(self) -> str:
        """Return the source of weight configuration for traceability."""
        return self._weight_source

    def set_weights(self, weights: np.ndarray):
        """Update weight configuration."""
        self.weights = weights
        if not np.isclose(np.sum(self.weights), 1.0):
            self.weights = self.weights / np.sum(self.weights)

    def compute_comprehensive_score(self, normalized: NormalizedMetrics) -> float:
        """Compute comprehensive score using weighted sum."""
        metrics_array = normalized.to_array()
        return float(np.dot(self.weights, metrics_array))

    def compute_score_decomposition(self, normalized: NormalizedMetrics) -> Dict[str, float]:
        """Compute contribution of each dimension to final score."""
        metrics_array = normalized.to_array()
        contributions = self.weights * metrics_array

        return {
            'communication': contributions[0],
            'rounds': contributions[1],
            'computation': contributions[2],
            'security': contributions[3],
            'total': float(np.sum(contributions))
        }

    def compute_all_scores(self, normalized_dict: Dict[str, NormalizedMetrics]) -> Dict[str, float]:
        """Compute comprehensive scores for all protocols."""
        return {
            name: self.compute_comprehensive_score(metrics)
            for name, metrics in normalized_dict.items()
        }

    def rank_protocols(self, scores: Dict[str, float]) -> List[Tuple[int, str, float]]:
        """Rank protocols by comprehensive score."""
        sorted_protocols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(i+1, name, score) for i, (name, score) in enumerate(sorted_protocols)]


class WeightSensitivityAnalyzer:
    """Analyzer for weight sensitivity in comprehensive scoring."""

    def __init__(self, weight_configs: Dict[str, np.ndarray]):
        """Initialize with multiple weight configurations."""
        self.weight_configs = weight_configs
        self.scoring_models = {
            name: ScoringModel(weights)
            for name, weights in weight_configs.items()
        }

    def analyze_sensitivity(self, normalized_dict: Dict[str, NormalizedMetrics]) -> Dict[str, Dict[str, float]]:
        """Analyze score sensitivity across weight configurations."""
        results = {}
        protocols = list(normalized_dict.keys())

        for protocol in protocols:
            results[protocol] = {}
            for config_name, model in self.scoring_models.items():
                score = model.compute_comprehensive_score(normalized_dict[protocol])
                results[protocol][config_name] = score

        return results

    def check_rank_stability(self, sensitivity_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, int]]:
        """Check ranking stability across weight configurations."""
        config_names = list(self.weight_configs.keys())
        rankings = {}

        for config_name in config_names:
            scores = {p: sensitivity_results[p][config_name]
                     for p in sensitivity_results.keys()}
            sorted_protocols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            rankings[config_name] = {
                name: rank + 1
                for rank, (name, _) in enumerate(sorted_protocols)
            }

        return rankings


class OLEEfficiencyCalculator:
    """Calculator for OLE efficiency ratio compared to OT baselines."""

    def __init__(self, reference_protocol: str = 'RLWE'):
        self.reference_protocol = reference_protocol

    def compute_efficiency_ratio(self, ole_metrics: Dict[str, float],
                                  reference_metrics: Dict[str, float]) -> float:
        """Compute efficiency ratio relative to reference."""
        comm_ratio = reference_metrics['communication'] / ole_metrics['communication']
        comp_ratio = reference_metrics['computation'] / ole_metrics['computation']

        raw_efficiency = np.sqrt(comm_ratio * comp_ratio)

        efficiency = min(1.0, raw_efficiency / 2.0)

        return efficiency

    def compute_all_efficiency_ratios(self, ole_data: Dict[str, Dict[str, float]],
                                       reference_data: Dict[str, float]) -> Dict[str, float]:
        """Compute efficiency ratios for all OLE protocols."""
        ratios = {}
        for name, metrics in ole_data.items():
            ratio = self.compute_efficiency_ratio(metrics, reference_data)
            ratios[name] = ratio

        if self.reference_protocol in ratios:
            ref_ratio = ratios[self.reference_protocol]
            for name in ratios:
                ratios[name] = ratios[name] / ref_ratio

        return ratios
