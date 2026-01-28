# -*- coding: utf-8 -*-
"""Metric Collector Module for aggregating performance metrics from multiple measurements."""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.real_measurement import RealMeasurement


@dataclass
class AggregatedMetrics:
    """Aggregated performance metrics."""
    protocol_name: str
    num_instances: int
    n_measurements: int

    # Communication statistics
    communication_mean: float
    communication_std: float
    communication_median: float
    communication_ci_lower: float  # 95% CI lower bound
    communication_ci_upper: float  # 95% CI upper bound

    # Computation statistics
    computation_mean: float
    computation_std: float
    computation_median: float
    computation_ci_lower: float
    computation_ci_upper: float

    # Total time statistics
    total_time_mean: float
    total_time_std: float

    # Memory statistics
    memory_mean: float
    memory_std: float

    # Coefficient of Variation (CV)
    communication_cv: float
    computation_cv: float

    def get_summary_dict(self) -> Dict:
        """Return summary dictionary."""
        return {
            'protocol': self.protocol_name,
            'n': self.num_instances,
            'comm_kb': f"{self.communication_mean:.2f} ± {self.communication_std:.2f}",
            'comp_ms': f"{self.computation_mean:.2f} ± {self.computation_std:.2f}",
            'total_ms': f"{self.total_time_mean:.2f} ± {self.total_time_std:.2f}",
            'comm_cv': f"{self.communication_cv:.1%}",
            'comp_cv': f"{self.computation_cv:.1%}"
        }


class MetricsCollector:
    """Performance metrics collector for aggregating data and analyzing noise."""

    def __init__(self, confidence_level: float = 0.95):
        """Initialize collector with confidence level."""
        self.confidence_level = confidence_level
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)
        self.measurements: Dict[str, Dict[int, List[RealMeasurement]]] = {}

    def add_measurements(self, measurements: List[RealMeasurement]):
        """Add measurement data."""
        for m in measurements:
            protocol = m.protocol_name
            n = m.num_instances

            if protocol not in self.measurements:
                self.measurements[protocol] = {}
            if n not in self.measurements[protocol]:
                self.measurements[protocol][n] = []

            self.measurements[protocol][n].append(m)

    def aggregate(self, protocol: str, num_instances: int,
                  remove_outliers: bool = True) -> AggregatedMetrics:
        """Aggregate measurements for specific protocol and size."""
        if protocol not in self.measurements:
            raise ValueError(f"No measurements found for protocol {protocol}")
        if num_instances not in self.measurements[protocol]:
            raise ValueError(f"No measurements found for size {num_instances}")

        measurements = self.measurements[protocol][num_instances]

        # Extract raw data
        comm_values = np.array([m.communication_kb for m in measurements])
        comp_values = np.array([m.computation_time_ms for m in measurements])
        total_values = np.array([m.total_time_ms for m in measurements])
        memory_values = np.array([m.memory_peak_kb for m in measurements])

        # Remove outliers using IQR
        if remove_outliers and len(comm_values) > 5:
            comm_values = self._remove_outliers(comm_values)
            comp_values = self._remove_outliers(comp_values)
            total_values = self._remove_outliers(total_values)
            memory_values = self._remove_outliers(memory_values)

        n = len(comm_values)

        # Compute statistics
        comm_mean = np.mean(comm_values)
        comm_std = np.std(comm_values, ddof=1) if n > 1 else 0
        comm_median = np.median(comm_values)
        comm_se = comm_std / np.sqrt(n) if n > 0 else 0

        comp_mean = np.mean(comp_values)
        comp_std = np.std(comp_values, ddof=1) if n > 1 else 0
        comp_median = np.median(comp_values)
        comp_se = comp_std / np.sqrt(n) if n > 0 else 0

        total_mean = np.mean(total_values)
        total_std = np.std(total_values, ddof=1) if n > 1 else 0

        memory_mean = np.mean(memory_values)
        memory_std = np.std(memory_values, ddof=1) if n > 1 else 0

        # Compute CI
        comm_ci = self.z_score * comm_se
        comp_ci = self.z_score * comp_se

        # Compute CV
        comm_cv = comm_std / comm_mean if comm_mean > 0 else 0
        comp_cv = comp_std / comp_mean if comp_mean > 0 else 0

        return AggregatedMetrics(
            protocol_name=protocol,
            num_instances=num_instances,
            n_measurements=n,
            communication_mean=comm_mean,
            communication_std=comm_std,
            communication_median=comm_median,
            communication_ci_lower=comm_mean - comm_ci,
            communication_ci_upper=comm_mean + comm_ci,
            computation_mean=comp_mean,
            computation_std=comp_std,
            computation_median=comp_median,
            computation_ci_lower=comp_mean - comp_ci,
            computation_ci_upper=comp_mean + comp_ci,
            total_time_mean=total_mean,
            total_time_std=total_std,
            memory_mean=memory_mean,
            memory_std=memory_std,
            communication_cv=comm_cv,
            computation_cv=comp_cv
        )

    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers using IQR method."""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return data[(data >= lower_bound) & (data <= upper_bound)]

    def aggregate_all(self, remove_outliers: bool = True) -> Dict[str, Dict[int, AggregatedMetrics]]:
        """Aggregate all measurement data."""
        results = {}
        for protocol in self.measurements:
            results[protocol] = {}
            for num_instances in self.measurements[protocol]:
                results[protocol][num_instances] = self.aggregate(
                    protocol, num_instances, remove_outliers
                )
        return results

    def get_variance_model(self) -> Dict[str, Dict[str, float]]:
        """Get measurement variance model for simulator calibration."""
        variance_model = {}

        for protocol in self.measurements:
            cv_comm_list = []
            cv_comp_list = []

            for num_instances in self.measurements[protocol]:
                metrics = self.aggregate(protocol, num_instances, remove_outliers=True)
                cv_comm_list.append(metrics.communication_cv)
                cv_comp_list.append(metrics.computation_cv)

            variance_model[protocol] = {
                'communication_cv_mean': np.mean(cv_comm_list),
                'communication_cv_std': np.std(cv_comm_list),
                'computation_cv_mean': np.mean(cv_comp_list),
                'computation_cv_std': np.std(cv_comp_list)
            }

        return variance_model


def compute_measurement_variance(measurements: List[RealMeasurement]) -> Dict[int, Dict[str, float]]:
    """Compute variance from repeated measurements."""
    # Group by size
    by_size = {}
    for m in measurements:
        n = m.num_instances
        if n not in by_size:
            by_size[n] = {'comm': [], 'comp': []}
        by_size[n]['comm'].append(m.communication_kb)
        by_size[n]['comp'].append(m.computation_time_ms)

    # Compute variance
    variance = {}
    for n, values in by_size.items():
        variance[n] = {
            'comm_mean': np.mean(values['comm']),
            'comm_std': np.std(values['comm'], ddof=1) if len(values['comm']) > 1 else 0,
            'comp_mean': np.mean(values['comp']),
            'comp_std': np.std(values['comp'], ddof=1) if len(values['comp']) > 1 else 0,
            'n_samples': len(values['comm'])
        }

    return variance


def interpolate_variance(variance_data: Dict[int, Dict[str, float]],
                         target_n: int) -> Dict[str, float]:
    """Interpolate variance for target scale."""
    sizes = sorted(variance_data.keys())

    if target_n in variance_data:
        return variance_data[target_n]

    if target_n < sizes[0]:
        return variance_data[sizes[0]]
    if target_n > sizes[-1]:
        return variance_data[sizes[-1]]

    # Find neighbors
    lower = max(s for s in sizes if s < target_n)
    upper = min(s for s in sizes if s > target_n)

    # Log-linear interpolation
    ratio = (np.log(target_n) - np.log(lower)) / (np.log(upper) - np.log(lower))

    interpolated = {}
    for key in ['comm_mean', 'comm_std', 'comp_mean', 'comp_std']:
        lower_val = variance_data[lower][key]
        upper_val = variance_data[upper][key]
        interpolated[key] = lower_val + ratio * (upper_val - lower_val)

    interpolated['n_samples'] = 0
    return interpolated
