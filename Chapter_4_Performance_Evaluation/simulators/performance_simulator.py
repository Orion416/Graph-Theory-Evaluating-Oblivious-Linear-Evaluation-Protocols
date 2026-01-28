# -*- coding: utf-8 -*-
"""
Performance simulator for OLE Protocol evaluation.

This module simulates protocol execution to generate SYNTHETIC performance
data based on theoretical complexity models with simulated measurement noise.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.protocol_models import OLEProtocolFamily, compute_ot_baseline_metrics
from config.experiment_config import ExperimentConfig, NetworkCondition


@dataclass
class SimulationResult:
    """Container for simulation results."""
    protocol_name: str
    scale: int
    n_instances: int
    n_trials: int

    # Communication results
    communication_mean: float
    communication_std: float
    communication_ci_lower: float
    communication_ci_upper: float

    # Computation results
    computation_mean: float
    computation_std: float
    computation_ci_lower: float
    computation_ci_upper: float

    # Round complexity
    setup_rounds: int
    online_rounds: int
    total_rounds: int

    # Security
    security_margin: float

    # Raw data for further analysis
    communication_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    computation_samples: np.ndarray = field(default_factory=lambda: np.array([]))


class PerformanceSimulator:
    """Simulator for OLE protocol performance evaluation."""

    def __init__(self, random_seed: int = 42):
        self.protocol_family = OLEProtocolFamily(random_seed)
        self.config = ExperimentConfig()
        self.rng = np.random.default_rng(random_seed)

    def simulate_single_protocol(self, protocol_name: str, scale: int,
                                  n_trials: int = 100) -> SimulationResult:
        """Simulate performance measurements for a single protocol."""
        model = self.protocol_family.get_model(protocol_name)
        if model is None:
            raise ValueError(f"Unknown protocol: {protocol_name}")

        n_instances = 2 ** scale

        # Simulate measurements
        samples = model.simulate_measurements(n_instances, n_trials)

        comm_samples = samples['communication']
        comp_samples = samples['computation']

        # Compute statistics
        comm_mean = np.mean(comm_samples)
        comm_std = np.std(comm_samples, ddof=1)
        comp_mean = np.mean(comp_samples)
        comp_std = np.std(comp_samples, ddof=1)

        # Compute confidence intervals
        z_score = self.config.get_z_score()
        comm_ci = z_score * comm_std / np.sqrt(n_trials)
        comp_ci = z_score * comp_std / np.sqrt(n_trials)

        # Get round complexity
        setup, online, total = model.get_round_complexity()

        return SimulationResult(
            protocol_name=protocol_name,
            scale=scale,
            n_instances=n_instances,
            n_trials=n_trials,
            communication_mean=comm_mean,
            communication_std=comm_std,
            communication_ci_lower=comm_mean - comm_ci,
            communication_ci_upper=comm_mean + comm_ci,
            computation_mean=comp_mean,
            computation_std=comp_std,
            computation_ci_lower=comp_mean - comp_ci,
            computation_ci_upper=comp_mean + comp_ci,
            setup_rounds=setup,
            online_rounds=online,
            total_rounds=total,
            security_margin=model.get_security_margin(),
            communication_samples=comm_samples,
            computation_samples=comp_samples
        )

    def simulate_all_protocols(self, scales: List[int] = None,
                                n_trials: int = 100) -> Dict[str, Dict[int, SimulationResult]]:
        """Simulate all protocols across multiple scales."""
        if scales is None:
            scales = self.config.problem_scales

        results = {}
        protocols = self.protocol_family.get_protocol_names()

        for protocol in protocols:
            results[protocol] = {}
            for scale in scales:
                result = self.simulate_single_protocol(protocol, scale, n_trials)
                results[protocol][scale] = result

        return results

    def generate_table9_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate communication complexity data for Table 9."""
        scales = self.config.problem_scales
        results = self.simulate_all_protocols(scales)

        table_data = {}
        for protocol in self.protocol_family.get_protocol_names():
            means = []
            stds = []
            for scale in scales:
                result = results[protocol][scale]
                means.append(result.communication_mean)
                stds.append(result.communication_std)
            table_data[protocol] = {
                'mean': np.array(means),
                'std': np.array(stds)
            }

        return table_data

    def generate_table10_data(self) -> Dict[str, Dict[str, int]]:
        """Generate round complexity data for Table 10."""
        return self.protocol_family.get_round_data()

    def generate_table11_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate computational overhead data for Table 11."""
        scales = self.config.problem_scales
        results = self.simulate_all_protocols(scales)

        table_data = {}
        for protocol in self.protocol_family.get_protocol_names():
            means = []
            stds = []
            for scale in scales:
                result = results[protocol][scale]
                means.append(result.computation_mean)
                stds.append(result.computation_std)
            table_data[protocol] = {
                'mean': np.array(means),
                'std': np.array(stds)
            }

        return table_data

    def generate_table15_data(self, scale: int = None) -> Dict[str, Dict[str, float]]:
        """Generate OLE vs OT comparison data for Table 15."""
        # Fix hardcoding: Get reference scale from config
        if scale is None:
            scale = self.config.get_reference_scale()
        n_instances = 2 ** scale

        # Get OLE protocol data
        ole_data = {}
        for protocol in self.protocol_family.get_protocol_names():
            result = self.simulate_single_protocol(protocol, scale, n_trials=100)
            ole_data[f"{protocol}-OLE"] = {
                'communication': result.communication_mean,
                'computation': result.computation_mean,
                'rounds': result.total_rounds,
                'type': 'OLE'
            }

        # Get OT baseline data
        ot_data = compute_ot_baseline_metrics(n_instances)
        for name, metrics in ot_data.items():
            ole_data[name] = {
                'communication': metrics['communication_kb'],
                'computation': metrics['computation_ops'],
                'rounds': metrics['rounds'],
                'type': 'OT'
            }

        return ole_data

    def simulate_application_performance(self, scale: int = None) -> Dict[str, Dict[str, float]]:
        """Simulate application-level performance for Table 17."""
        # Fix hardcoding: Get reference scale from config
        if scale is None:
            scale = self.config.get_reference_scale()

        n_instances = 2 ** scale
        results = {}

        # Fix hardcoding: Get network params from config
        app_config = self.config.get_application_simulation_config()
        bandwidth_mbps = app_config.get('reference_bandwidth_mbps', 100.0)
        latency_ms = app_config.get('reference_latency_ms', 5.0)
        exec_variance = app_config.get('execution_variance', 0.05)

        bandwidth_kbps = bandwidth_mbps * 1000 / 8  # KB/s
        latency_s = latency_ms / 1000  # seconds

        for protocol in self.protocol_family.get_protocol_names():
            result = self.simulate_single_protocol(protocol, scale, n_trials=100)

            # Communication time
            comm_time = result.communication_mean / bandwidth_kbps

            # Round trip time contribution
            rtt_time = result.total_rounds * 2 * latency_s

            # Computation time (convert ops to seconds, assuming 1M ops/s)
            comp_time = result.computation_mean

            # Total execution time
            exec_time = comm_time + rtt_time + comp_time

            # Add realistic variance from config
            exec_std = exec_time * exec_variance

            results[protocol] = {
                'time_mean': exec_time,
                'time_std': exec_std
            }

        return results

    def simulate_network_conditions(self, conditions: List[NetworkCondition] = None,
                                      scale: int = None) -> Dict[str, Dict[str, List[str]]]:
        """Simulate rankings under different network conditions for Table 18."""
        if conditions is None:
            conditions = self.config.network_conditions

        # Fix hardcoding: Get reference scale from config
        if scale is None:
            scale = self.config.get_reference_scale()

        n_instances = 2 ** scale

        results = {}

        for condition in conditions:
            # Model effective performance under network condition
            protocol_scores = {}

            for protocol in self.protocol_family.get_protocol_names():
                sim_result = self.simulate_single_protocol(protocol, scale)

                # Network-adjusted performance score
                bandwidth_factor = np.log10(condition.bandwidth_mbps + 1) / 3
                latency_factor = np.log10(condition.latency_ms + 1) / 2

                # Communication cost weighted by bandwidth scarcity
                comm_cost = sim_result.communication_mean * (1 - bandwidth_factor)

                # Round cost weighted by latency
                round_cost = sim_result.total_rounds * latency_factor * 1000

                # Computation cost
                comp_cost = sim_result.computation_mean * 10

                # Combined score (lower is better)
                total_cost = comm_cost + round_cost + comp_cost
                protocol_scores[protocol] = 1000 / total_cost

            # Rank protocols
            sorted_protocols = sorted(protocol_scores.items(), key=lambda x: x[1], reverse=True)
            top_3 = [p[0] for p in sorted_protocols[:3]]

            results[condition.name] = {
                'top_3': top_3,
                'bandwidth': condition.bandwidth_mbps,
                'latency': condition.latency_ms
            }

        return results
