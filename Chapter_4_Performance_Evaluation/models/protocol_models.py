# -*- coding: utf-8 -*-
"""OLE protocol mathematical models module."""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import interp1d
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.protocol_config import ProtocolConfig, ProtocolParameters
from config.experiment_config import load_yaml_config, ExperimentConfig


@dataclass
class ProtocolMetrics:
    """Container for protocol performance metrics."""
    communication_kb: float
    communication_std: float
    computation_ops: float
    computation_std: float
    setup_rounds: int
    online_rounds: int
    total_rounds: int
    security_margin: float


class TheoreticalComplexityModel:
    """Theoretical complexity model derived from cryptographic parameters."""

    def __init__(self, protocol_name: str, params: ProtocolParameters,
                 constants_file: str = "complexity_constants.yaml"):
        """Initialize complexity model with protocol parameters."""
        self.protocol_name = protocol_name
        self.params = params
        self._constants_file = constants_file
        self._load_constants()
        self._compute_complexity_functions()

    def _load_constants(self):
        """Load complexity constants from YAML configuration."""
        self._constants = load_yaml_config(self._constants_file)

        self.AES_OPS_PER_HASH = self._get_constant('AES_OPS_PER_HASH')
        self.AES_OPS_PER_NTT_BUTTERFLY = self._get_constant('AES_OPS_PER_NTT_BUTTERFLY')
        self.AES_OPS_PER_EC_MUL = self._get_constant('AES_OPS_PER_EC_MUL')
        self.AES_OPS_PER_HE_MULT = self._get_constant('AES_OPS_PER_HE_MULT')

        self.RLWE_PER_OLE_FACTOR = self._get_constant('RLWE_PER_OLE_FACTOR')
        self.NOISY_ENCODING_OVERHEAD = self._get_constant('NOISY_ENCODING_OVERHEAD')
        self.NOISY_POLYNOMIAL_DEGREE = self._get_constant('NOISY_POLYNOMIAL_DEGREE')
        self.PCG_COMPRESSION_FACTOR = self._get_constant('PCG_COMPRESSION_FACTOR')
        self.PCG_COMPUTATION_FACTOR = self._get_constant('PCG_COMPUTATION_FACTOR')
        self.HE_PACKING_OVERHEAD = self._get_constant('HE_PACKING_OVERHEAD')

        self.RLWE_PER_OLE_COMP_FACTOR = self._get_constant('RLWE_PER_OLE_COMP_FACTOR')
        self.LPN_OVERHEAD_FACTOR = self._get_constant('LPN_OVERHEAD_FACTOR')
        self.HE_BASE_COMP_FACTOR = self._get_constant('HE_BASE_COMP_FACTOR')
        self.SECURITY_PARAMETER_DEFAULT = self._get_constant('SECURITY_PARAMETER_DEFAULT')

        self.COMM_NOISE_STD = self._get_constant('COMMUNICATION_NOISE_STD')
        self.COMP_NOISE_STD = self._get_constant('COMPUTATION_NOISE_STD')

        config = ExperimentConfig()
        self.REFERENCE_SCALES = np.array(config.problem_scales)

    def _get_constant(self, name: str) -> float:
        """Get constant value with fallback."""
        entry = self._constants.get(name, {})
        if isinstance(entry, dict):
            return entry.get('value', 0)
        return entry

    def get_constant_reference(self, constant_name: str) -> str:
        """Get the academic reference for a specific constant."""
        if constant_name in self._constants:
            entry = self._constants[constant_name]
            if isinstance(entry, dict):
                return entry.get('reference', 'No reference available')
        return 'Unknown constant'

    def get_constant_rationale(self, constant_name: str) -> str:
        """Get the rationale/explanation for a specific constant."""
        if constant_name in self._constants:
            entry = self._constants[constant_name]
            if isinstance(entry, dict):
                return entry.get('rationale', '')
        return ''

    def _compute_complexity_functions(self):
        """Compute complexity based on theoretical analysis of each protocol."""
        N_values = 2 ** self.REFERENCE_SCALES

        if self.protocol_name == 'RLWE':
            self._compute_rlwe_complexity(N_values)
        elif self.protocol_name == 'IKNP':
            self._compute_iknp_complexity(N_values)
        elif self.protocol_name == 'Noisy':
            self._compute_noisy_complexity(N_values)
        elif self.protocol_name == 'PCG':
            self._compute_pcg_complexity(N_values)
        elif self.protocol_name == 'HE':
            self._compute_he_complexity(N_values)

        log_n = np.log2(N_values)
        self._comm_interp = interp1d(log_n, self.comm_values,
                                      kind='cubic', fill_value='extrapolate')
        self._comp_interp = interp1d(log_n, self.comp_values,
                                      kind='cubic', fill_value='extrapolate')

    def _compute_rlwe_complexity(self, N_values):
        """RLWE-OLE complexity from Ring-LWE parameters."""
        n = self.params.ring_dimension
        log_q = self.params.modulus_bits
        k = self.params.field_bits

        bytes_per_ring = n * np.ceil(log_q / 8)
        ring_kb = bytes_per_ring / 1024

        base_comm = 2 * ring_kb

        per_ole_comm = k / 8 / 1024 / n * self.RLWE_PER_OLE_FACTOR

        self.comm_values = base_comm + per_ole_comm * N_values

        ntt_ops = n * np.log2(n) * self.AES_OPS_PER_NTT_BUTTERFLY / 1e6
        num_ntts = 4

        base_comp = ntt_ops * num_ntts

        per_ole_comp = k / n * self.RLWE_PER_OLE_COMP_FACTOR

        self.comp_values = base_comp + per_ole_comp * N_values

    def _compute_iknp_complexity(self, N_values):
        """IKNP-OLE complexity from OT extension parameters."""
        k = self.params.field_bits

        comm_per_ole = k * 2 / 8 / 1024

        self.comm_values = comm_per_ole * N_values

        hash_ops_per_ole = k * self.AES_OPS_PER_HASH / 1e6

        self.comp_values = hash_ops_per_ole * N_values

    def _compute_noisy_complexity(self, N_values):
        """Noisy-Encoding OLE complexity."""
        k = self.params.field_bits
        d = self.NOISY_POLYNOMIAL_DEGREE

        comm_per_ole = k * d * 2 / 8 / 1024 * self.NOISY_ENCODING_OVERHEAD

        self.comm_values = comm_per_ole * N_values

        poly_ops_per_ole = k * d * 1.5 * self.AES_OPS_PER_HASH / 1e6

        self.comp_values = poly_ops_per_ole * N_values

    def _compute_pcg_complexity(self, N_values):
        """PCG-OLE complexity from pseudorandom correlation generation."""
        k = self.params.field_bits
        lambda_sec = int(self.SECURITY_PARAMETER_DEFAULT)

        seed_kb = lambda_sec / 8 / 1024
        expansion_overhead = np.sqrt(N_values) * k / 8 / 1024 * self.PCG_COMPRESSION_FACTOR

        self.comm_values = seed_kb + expansion_overhead

        prg_ops_per_output = np.log2(N_values) * 2
        total_comp = N_values * prg_ops_per_output / 1e6

        lpn_overhead = np.sqrt(N_values) * k * self.LPN_OVERHEAD_FACTOR

        self.comp_values = total_comp * self.PCG_COMPUTATION_FACTOR + lpn_overhead

    def _compute_he_complexity(self, N_values):
        """HE-OLE complexity from homomorphic encryption parameters."""
        n = self.params.ring_dimension
        log_q = self.params.modulus_bits
        k = self.params.field_bits

        bytes_per_ct = 2 * n * np.ceil(log_q / 8)
        ct_kb = bytes_per_ct / 1024

        num_cts = np.ceil(N_values / n)

        self.comm_values = 2 * ct_kb + num_cts * k / 8 / 1024 * self.HE_PACKING_OVERHEAD

        he_mult_per_ct = self.AES_OPS_PER_HE_MULT / 1e6

        base_comp = n * np.log2(n) * self.HE_BASE_COMP_FACTOR

        per_ct_comp = he_mult_per_ct * 2

        self.comp_values = base_comp + per_ct_comp * num_cts

    def compute_communication(self, n_instances: int) -> Tuple[float, float]:
        """Compute communication complexity with measurement noise."""
        log_n = np.log2(n_instances)
        mean = float(self._comm_interp(log_n))
        mean = max(mean, 1.0)

        std = mean * self.COMM_NOISE_STD

        return mean, std

    def compute_computation(self, n_instances: int) -> Tuple[float, float]:
        """Compute computational overhead with measurement noise."""
        log_n = np.log2(n_instances)
        mean = float(self._comp_interp(log_n))
        mean = max(mean, 0.1)

        std = mean * self.COMP_NOISE_STD

        return mean, std


class ProtocolModel:
    """Mathematical model for computing protocol complexity metrics."""

    def __init__(self, params: ProtocolParameters, random_seed: Optional[int] = None):
        """Initialize protocol model."""
        self.params = params
        self.rng = np.random.default_rng(random_seed)
        self.complexity_model = TheoreticalComplexityModel(params.short_name, params)

    def compute_communication_complexity(self, n_instances: int) -> Tuple[float, float]:
        """Compute communication complexity in KB."""
        return self.complexity_model.compute_communication(n_instances)

    def compute_computational_overhead(self, n_instances: int) -> Tuple[float, float]:
        """Compute computational overhead in millions of symmetric operations."""
        return self.complexity_model.compute_computation(n_instances)

    def get_round_complexity(self) -> Tuple[int, int, int]:
        """Get round complexity metrics."""
        return (
            self.params.setup_rounds,
            self.params.online_rounds,
            self.params.setup_rounds + self.params.online_rounds
        )

    def get_security_margin(self) -> float:
        """Get security margin based on cryptographic parameters."""
        return self.params.security_margin

    def compute_metrics(self, n_instances: int) -> ProtocolMetrics:
        """Compute all performance metrics for given problem scale."""
        comm_mean, comm_std = self.compute_communication_complexity(n_instances)
        comp_mean, comp_std = self.compute_computational_overhead(n_instances)
        setup, online, total = self.get_round_complexity()

        return ProtocolMetrics(
            communication_kb=comm_mean,
            communication_std=comm_std,
            computation_ops=comp_mean,
            computation_std=comp_std,
            setup_rounds=setup,
            online_rounds=online,
            total_rounds=total,
            security_margin=self.get_security_margin()
        )

    def simulate_measurements(self, n_instances: int, n_trials: int) -> Dict[str, np.ndarray]:
        """Simulate multiple measurement trials with realistic noise."""
        comm_mean, comm_std = self.compute_communication_complexity(n_instances)
        comp_mean, comp_std = self.compute_computational_overhead(n_instances)

        comm_samples = self.rng.lognormal(
            mean=np.log(comm_mean) - 0.5 * (comm_std/comm_mean)**2,
            sigma=comm_std/comm_mean,
            size=n_trials
        )

        comp_samples = self.rng.lognormal(
            mean=np.log(comp_mean) - 0.5 * (comp_std/comp_mean)**2,
            sigma=comp_std/comp_mean,
            size=n_trials
        )

        return {
            'communication': comm_samples,
            'computation': comp_samples
        }


class OLEProtocolFamily:
    """Manager for the family of OLE protocols."""

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize protocol family."""
        if random_seed is None:
            config = ExperimentConfig()
            seed_config = config.get_random_seed_config()
            if seed_config.get('mode') == 'time_based':
                import time
                random_seed = int(time.time())
            else:
                random_seed = seed_config.get('fixed_value', 42)

        self._random_seed = random_seed
        self.config = ProtocolConfig()
        self.models: Dict[str, ProtocolModel] = {}
        self._init_models()

    def _init_models(self):
        """Initialize protocol models."""
        for name, params in self.config.get_all_protocols().items():
            protocol_seed = self._random_seed + hash(name) % 10000
            self.models[name] = ProtocolModel(params, protocol_seed)

    def get_model(self, name: str) -> Optional[ProtocolModel]:
        """Get protocol model by name."""
        return self.models.get(name)

    def get_protocol_names(self) -> list:
        """Get list of protocol names."""
        return list(self.models.keys())

    def compute_all_metrics(self, n_instances: int) -> Dict[str, ProtocolMetrics]:
        """Compute metrics for all protocols at given scale."""
        return {
            name: model.compute_metrics(n_instances)
            for name, model in self.models.items()
        }

    def compute_metrics_over_scales(self, scales: list) -> Dict[str, Dict[int, ProtocolMetrics]]:
        """Compute metrics for all protocols across multiple scales."""
        results = {}
        for name in self.get_protocol_names():
            results[name] = {}
            for scale in scales:
                n_instances = 2 ** scale
                results[name][scale] = self.models[name].compute_metrics(n_instances)
        return results

    def get_communication_data(self, scales: list) -> Dict[str, Dict[str, np.ndarray]]:
        """Get communication complexity data for plotting."""
        data = {}
        for name in self.get_protocol_names():
            means = []
            stds = []
            for scale in scales:
                n_instances = 2 ** scale
                mean, std = self.models[name].compute_communication_complexity(n_instances)
                means.append(mean)
                stds.append(std)
            data[name] = {
                'mean': np.array(means),
                'std': np.array(stds)
            }
        return data

    def get_computation_data(self, scales: list) -> Dict[str, Dict[str, np.ndarray]]:
        """Get computation overhead data for plotting."""
        data = {}
        for name in self.get_protocol_names():
            means = []
            stds = []
            for scale in scales:
                n_instances = 2 ** scale
                mean, std = self.models[name].compute_computational_overhead(n_instances)
                means.append(mean)
                stds.append(std)
            data[name] = {
                'mean': np.array(means),
                'std': np.array(stds)
            }
        return data

    def get_round_data(self) -> Dict[str, Dict[str, int]]:
        """Get round complexity data."""
        data = {}
        for name in self.get_protocol_names():
            setup, online, total = self.models[name].get_round_complexity()
            data[name] = {
                'setup': setup,
                'online': online,
                'total': total
            }
        return data

    def get_random_seed(self) -> int:
        """Get the random seed used for this family."""
        return self._random_seed


def compute_ot_baseline_metrics(n_instances: int) -> Dict[str, Dict[str, float]]:
    """Compute OT baseline metrics for comparison."""
    constants = load_yaml_config('complexity_constants.yaml')
    aes_per_hash = constants.get('AES_OPS_PER_HASH', {}).get('value', 100)
    aes_per_ec_mul = constants.get('AES_OPS_PER_EC_MUL', {}).get('value', 2000)

    baselines = load_yaml_config('ot_baselines.yaml')
    common = baselines.get('common_parameters', {})
    field_bits = common.get('field_bits', 128)
    lambda_sec = int(constants.get('SECURITY_PARAMETER_DEFAULT', {}).get('value', 128))

    results = {}

    simplest = baselines.get('Simplest_OT', {})
    ec_point_bytes = common.get('ec_point_bytes', 32)
    comm_per_ot = simplest.get('comm_per_ot_bytes', 2 * ec_point_bytes)
    comm_per_ole_simplest = field_bits * comm_per_ot / 1024

    comp_per_ot_simplest = simplest.get('comp_per_ot_aes_equiv', 2 * aes_per_ec_mul) / 1e6
    comp_per_ole_simplest = field_bits * comp_per_ot_simplest

    results['Simplest OT'] = {
        'communication_kb': comm_per_ole_simplest * n_instances,
        'computation_ops': comp_per_ole_simplest * n_instances,
        'rounds': simplest.get('rounds', 2),
        'type': 'OT'
    }

    iknp = baselines.get('IKNP_OT', {})
    bits_per_extended_ot = iknp.get('bits_per_extended_ot', 64)
    comm_per_ole_iknp = field_bits * bits_per_extended_ot / 8 / 1024

    hash_ops_per_ot = iknp.get('hash_ops_per_ot', 2)
    comp_per_ole_iknp = field_bits * hash_ops_per_ot * aes_per_hash / 1e6

    results['IKNP OT'] = {
        'communication_kb': comm_per_ole_iknp * n_instances,
        'computation_ops': comp_per_ole_iknp * n_instances,
        'rounds': iknp.get('rounds', 4),
        'type': 'OT'
    }

    silent = baselines.get('Silent_OT', {})
    seed_kb = silent.get('seed_bits', lambda_sec) / 8 / 1024
    expansion_factor = np.sqrt(n_instances) * field_bits / 8 / 1024

    prg_ops = silent.get('prg_ops_per_output', 10)
    prg_ops_per_output = np.log2(n_instances) * prg_ops
    comp_silent = n_instances * prg_ops_per_output / 1e6

    results['Silent OT'] = {
        'communication_kb': seed_kb + expansion_factor * 0.3,
        'computation_ops': comp_silent * 0.1,
        'rounds': silent.get('rounds', 8),
        'type': 'OT'
    }

    return results


class CalibratedComplexityModel:
    """Calibrated complexity model using real measurements."""

    def __init__(self, protocol_name: str, params: ProtocolParameters,
                 calibration_data: Dict = None):
        """Initialize calibrated complexity model."""
        self.protocol_name = protocol_name
        self.params = params
        self.calibration = calibration_data or {}

        self.theoretical_model = TheoreticalComplexityModel(protocol_name, params)

        self._load_calibration_config()

    def _load_calibration_config(self):
        """Load calibration parameters from configuration."""
        try:
            config = load_yaml_config('benchmark_config.yaml')
            if config.get('calibration', {}).get('enabled', False):
                factors = config['calibration'].get('factors', {})
                if self.protocol_name in factors:
                    self.calibration.update(factors[self.protocol_name])
        except FileNotFoundError:
            pass

    def compute_communication(self, n_instances: int) -> Tuple[float, float]:
        """Compute calibrated communication complexity."""
        mean, std = self.theoretical_model.compute_communication(n_instances)

        correction = self.calibration.get('communication', 1.0)
        calibrated_mean = mean * correction

        calibrated_std = std * correction

        return calibrated_mean, calibrated_std

    def compute_computation(self, n_instances: int) -> Tuple[float, float]:
        """Compute calibrated computation complexity."""
        mean, std = self.theoretical_model.compute_computation(n_instances)

        correction = self.calibration.get('computation', 1.0)
        calibrated_mean = mean * correction
        calibrated_std = std * correction

        return calibrated_mean, calibrated_std

    def load_calibration_from_benchmark(self, benchmark_results: Dict):
        """Load calibration data from benchmark results."""
        if not benchmark_results:
            return

        comm_corrections = []
        comp_corrections = []

        for n_instances, measurements in benchmark_results.items():
            theo_comm, _ = self.theoretical_model.compute_communication(n_instances)
            theo_comp, _ = self.theoretical_model.compute_computation(n_instances)

            real_comm = measurements.get('communication_kb', theo_comm)
            real_comp = measurements.get('computation_time_ms', theo_comp)

            if theo_comm > 0:
                comm_corrections.append(real_comm / theo_comm)
            if theo_comp > 0:
                comp_corrections.append(real_comp / theo_comp)

        if comm_corrections:
            self.calibration['communication'] = np.median(comm_corrections)
        if comp_corrections:
            self.calibration['computation'] = np.median(comp_corrections)

    def get_calibration_factors(self) -> Dict[str, float]:
        """Get current calibration factors."""
        return self.calibration.copy()


class HybridComplexityModel:
    """Hybrid complexity model combining theory and measurements."""

    def __init__(self, theoretical_model: TheoreticalComplexityModel,
                 benchmark_data: Dict[int, Dict[str, float]] = None):
        """Initialize hybrid model."""
        self.theoretical = theoretical_model
        self.benchmark = benchmark_data or {}

        self._compute_calibration()

    def _compute_calibration(self):
        """Compute calibration factors from benchmark data."""
        self.calibration = {'communication': 1.0, 'computation': 1.0}

        if not self.benchmark:
            return

        comm_factors = []
        comp_factors = []

        for n, data in self.benchmark.items():
            theo_comm, _ = self.theoretical.compute_communication(n)
            theo_comp, _ = self.theoretical.compute_computation(n)

            if theo_comm > 0 and 'communication_kb' in data:
                comm_factors.append(data['communication_kb'] / theo_comm)
            if theo_comp > 0 and 'computation_time_ms' in data:
                comp_factors.append(data['computation_time_ms'] / theo_comp)

        if comm_factors:
            self.calibration['communication'] = np.median(comm_factors)
        if comp_factors:
            self.calibration['computation'] = np.median(comp_factors)

    def estimate(self, num_instances: int) -> Dict[str, float]:
        """Estimate performance metrics for given scale."""
        if num_instances in self.benchmark:
            return self.benchmark[num_instances].copy()

        measured_sizes = sorted(self.benchmark.keys()) if self.benchmark else []

        if not measured_sizes:
            comm, comm_std = self.theoretical.compute_communication(num_instances)
            comp, comp_std = self.theoretical.compute_computation(num_instances)
            return {
                'communication_kb': comm * self.calibration['communication'],
                'computation_time_ms': comp * self.calibration['computation'],
                'source': 'theoretical_calibrated'
            }

        lower = max([s for s in measured_sizes if s < num_instances], default=None)
        upper = min([s for s in measured_sizes if s > num_instances], default=None)

        if lower and upper:
            log_ratio = (np.log(num_instances) - np.log(lower)) / (np.log(upper) - np.log(lower))

            interpolated = {}
            for metric in self.benchmark[lower].keys():
                if isinstance(self.benchmark[lower][metric], (int, float)):
                    lower_val = self.benchmark[lower][metric]
                    upper_val = self.benchmark[upper][metric]
                    if lower_val > 0 and upper_val > 0:
                        interpolated[metric] = np.exp(
                            np.log(lower_val) * (1 - log_ratio) +
                            np.log(upper_val) * log_ratio
                        )
                    else:
                        interpolated[metric] = lower_val * (1 - log_ratio) + upper_val * log_ratio

            interpolated['source'] = 'interpolated'
            return interpolated

        comm, _ = self.theoretical.compute_communication(num_instances)
        comp, _ = self.theoretical.compute_computation(num_instances)

        return {
            'communication_kb': comm * self.calibration['communication'],
            'computation_time_ms': comp * self.calibration['computation'],
            'source': 'extrapolated'
        }

    def get_calibration_info(self) -> Dict:
        """Get calibration information."""
        return {
            'calibration_factors': self.calibration,
            'measured_points': list(self.benchmark.keys()),
            'model_type': 'hybrid'
        }
