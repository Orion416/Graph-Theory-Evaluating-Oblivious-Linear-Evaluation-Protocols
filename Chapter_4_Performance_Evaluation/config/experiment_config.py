# -*- coding: utf-8 -*-
"""
Experiment configuration for OLE Protocol Performance Evaluation.

This module loads experimental parameters from external YAML configuration
files to ensure transparency and traceability of all parameter choices.
"""

import numpy as np
import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    return Path(__file__).parent


def load_yaml_config(filename: str) -> dict:
    """Load configuration from YAML file with validation."""
    config_dir = get_config_dir()
    filepath = config_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {filepath}\n"
            f"Please ensure all configuration files are present in {config_dir}"
        )

    with open(filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


@dataclass
class NetworkCondition:
    """Network condition configuration for performance modeling."""
    name: str
    bandwidth_mbps: float
    latency_ms: float
    description: str = ""

    def __post_init__(self):
        """Validate network condition parameters."""
        if self.bandwidth_mbps <= 0:
            raise ValueError(f"Bandwidth must be positive, got {self.bandwidth_mbps}")
        if self.latency_ms < 0:
            raise ValueError(f"Latency cannot be negative, got {self.latency_ms}")


@dataclass
class WeightConfig:
    """Weight configuration for multi-dimensional evaluation."""
    name: str
    communication: float
    rounds: float
    computation: float
    security: float
    description: str = ""
    rationale: str = ""

    def __post_init__(self):
        """Validate weight configuration."""
        total = self.communication + self.rounds + self.computation + self.security
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Weights must sum to 1.0, got {total} for config '{self.name}'"
            )
        for name, val in [('communication', self.communication),
                          ('rounds', self.rounds),
                          ('computation', self.computation),
                          ('security', self.security)]:
            if not (0 <= val <= 1):
                raise ValueError(f"Weight {name} must be in [0,1], got {val}")

    def to_array(self) -> np.ndarray:
        """Convert weights to numpy array."""
        return np.array([
            self.communication,
            self.rounds,
            self.computation,
            self.security
        ])


class ExperimentConfig:
    """Configuration manager for experiments."""

    def __init__(self, config_file: str = "parameters.yaml"):
        """Initialize experiment configuration from YAML file."""
        self._config_file = config_file
        self._config = load_yaml_config(config_file)
        self._validate_config()
        self._load_parameters()

    def _validate_config(self):
        """Validate configuration structure and values."""
        required_sections = ['framework', 'experiment', 'network_conditions', 'weight_configs']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required section '{section}' in configuration")

        # Validate weight configurations sum to 1
        for name, weights in self._config['weight_configs'].items():
            total = sum([
                weights.get('communication', 0),
                weights.get('rounds', 0),
                weights.get('computation', 0),
                weights.get('security', 0)
            ])
            if not (0.99 <= total <= 1.01):
                raise ValueError(
                    f"Weights in '{name}' must sum to 1.0, got {total}"
                )

    def _load_parameters(self):
        """Load parameters from configuration."""
        exp = self._config['experiment']

        # Problem scales
        self.problem_scales = exp['problem_scales']

        # Security parameters
        self.security_parameter = exp['security_parameter']
        self.statistical_security = exp['statistical_security']

        # Simulation parameters
        self.num_trials = exp['num_trials']
        self.confidence_level = exp['confidence_level']

        # Load network conditions
        self.network_conditions = [
            NetworkCondition(
                name=nc['name'],
                bandwidth_mbps=nc['bandwidth_mbps'],
                latency_ms=nc['latency_ms'],
                description=nc.get('description', '')
            )
            for nc in self._config['network_conditions']
        ]

        # Load weight configurations
        self.weight_configs = self._load_weight_configs()

    def _load_weight_configs(self) -> Dict[str, WeightConfig]:
        """Load weight configurations from YAML."""
        configs = {}
        for name, wc in self._config['weight_configs'].items():
            configs[name] = WeightConfig(
                name=name,
                communication=wc['communication'],
                rounds=wc['rounds'],
                computation=wc['computation'],
                security=wc['security'],
                description=wc.get('description', ''),
                rationale=wc.get('rationale', '')
            )
        return configs

    def get_scale_values(self) -> np.ndarray:
        """Get actual problem scale values (2^N)."""
        return np.array([2**n for n in self.problem_scales])

    def get_z_score(self) -> float:
        """Get z-score for confidence interval calculation."""
        from scipy import stats
        return stats.norm.ppf((1 + self.confidence_level) / 2)

    def get_reference_scale_index(self) -> int:
        """Get index of reference scale (N=2^16) for detailed analysis."""
        try:
            return self.problem_scales.index(16)
        except ValueError:
            return len(self.problem_scales) // 2

    def get_config_source(self) -> str:
        """Return the source file for configuration traceability."""
        return f"config/{self._config_file}"

    def get_framework_disclaimer(self) -> str:
        """Return the framework disclaimer text."""
        return self._config['framework'].get('disclaimer', '')

    def get_framework_type(self) -> str:
        """Return the framework type (theoretical_model)."""
        return self._config['framework'].get('type', 'theoretical_model')

    def get_random_seed_config(self) -> dict:
        """Get random seed configuration."""
        return self._config.get('random_seed', {
            'mode': 'fixed',
            'fixed_value': 42
        })

    def get_hypothetical_environment(self) -> dict:
        """Get hypothetical environment configuration."""
        return self._config.get('hypothetical_environment', {})

    def get_application_simulation_params(self) -> dict:
        """Get application simulation parameters."""
        return self._config.get('application_simulation', {
            'reference_bandwidth_mbps': 100.0,
            'reference_latency_ms': 5.0,
            'execution_variance': 0.05
        })

    def get_application_simulation_config(self) -> dict:
        """Get application simulation configuration."""
        return self.get_application_simulation_params()

    def get_reference_scale(self) -> int:
        """Get the reference problem scale exponent for detailed analysis."""
        # Use 16 as reference scale if available
        if 16 in self.problem_scales:
            return 16
        # Otherwise use middle value
        return self.problem_scales[len(self.problem_scales) // 2]


def get_random_seed(seed_arg: Optional[str] = None) -> int:
    """Determine random seed based on configuration and optional argument."""
    config = ExperimentConfig()
    seed_config = config.get_random_seed_config()

    if seed_arg is not None:
        if seed_arg.lower() == 'time':
            import time
            return int(time.time())
        try:
            return int(seed_arg)
        except ValueError:
            pass

    mode = seed_config.get('mode', 'fixed')

    if mode == 'time_based':
        import time
        return int(time.time())

    return seed_config.get('fixed_value', 42)
