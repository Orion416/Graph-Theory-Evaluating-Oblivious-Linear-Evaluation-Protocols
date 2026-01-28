# -*- coding: utf-8 -*-
"""
Protocol configuration parameters for OLE Protocol Family.

This module loads cryptographic parameters for each OLE protocol variant
from external YAML configuration files.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from pathlib import Path

from config.experiment_config import load_yaml_config, get_config_dir


@dataclass
class ProtocolParameters:
    """Base parameters for OLE protocol variants."""
    name: str
    short_name: str

    # Cryptographic parameters
    ring_dimension: int
    modulus_bits: int
    field_bits: int

    # Protocol characteristics
    setup_rounds: int
    online_rounds: int

    # Security parameters
    security_bits: int
    security_margin: float

    # Protocol type for classification
    protocol_type: str
    color: str
    marker: str


def compute_security_margin(protocol_type: str, security_bits: int,
                            ring_dimension: int = 0, modulus_bits: int = 0,
                            constants: Optional[dict] = None) -> float:
    """Compute security margin based on cryptographic parameters."""
    if constants is None:
        constants = load_yaml_config('complexity_constants.yaml')

    base_security = 128

    if protocol_type == 'lattice':
        # Lattice security estimation
        if ring_dimension > 1 and modulus_bits > 0:
            estimated_security = ring_dimension * np.log2(ring_dimension) / modulus_bits * 4
            margin = min(1.0, estimated_security / base_security)
        else:
            margin = security_bits / base_security

    elif protocol_type == 'he_based':
        # HE uses larger parameters for stronger security
        if ring_dimension > 1 and modulus_bits > 0:
            estimated_security = ring_dimension * np.log2(ring_dimension) / modulus_bits * 5
            margin = min(1.0, estimated_security / base_security)
        else:
            margin = security_bits / base_security

    elif protocol_type == 'ot_based':
        # OT security from computational assumptions
        ot_margin_coeff = constants.get('SECURITY_MARGIN_OT', {}).get('value', 0.95)
        margin = security_bits / base_security * ot_margin_coeff

    elif protocol_type == 'lpn_based':
        # LPN security margin
        lpn_margin_coeff = constants.get('SECURITY_MARGIN_LPN', {}).get('value', 0.92)
        margin = security_bits / base_security * lpn_margin_coeff

    else:
        margin = security_bits / base_security

    return min(1.0, max(0.0, margin))


class ProtocolConfig:
    """Configuration manager for all OLE protocols."""

    def __init__(self, protocol_file: str = "protocols.yaml",
                 constants_file: str = "complexity_constants.yaml"):
        """Initialize protocol configuration from YAML files."""
        self._protocol_file = protocol_file
        self._constants_file = constants_file
        self._config = load_yaml_config(protocol_file)
        self._constants = load_yaml_config(constants_file)
        self._protocols = self._initialize_protocols()

    def _initialize_protocols(self) -> Dict[str, ProtocolParameters]:
        """Initialize protocol parameters from YAML configuration."""
        protocols = {}

        for name, params in self._config.items():
            # Validate required fields
            required_fields = ['name', 'short_name', 'protocol_type',
                              'ring_dimension', 'modulus_bits', 'field_bits',
                              'setup_rounds', 'online_rounds', 'security_bits',
                              'color', 'marker']
            for field in required_fields:
                if field not in params:
                    raise ValueError(
                        f"Protocol '{name}' missing required field '{field}'"
                    )

            # Compute security margin
            security_margin = compute_security_margin(
                params['protocol_type'],
                params['security_bits'],
                params['ring_dimension'],
                params['modulus_bits'],
                self._constants
            )

            # Apply UC bonus for Noisy-Encoding
            if name == 'Noisy':
                uc_bonus = self._constants.get('SECURITY_MARGIN_UC_BONUS', {}).get('value', 1.02)
                security_margin = min(1.0, security_margin * uc_bonus)

            protocols[name] = ProtocolParameters(
                name=params['name'],
                short_name=params['short_name'],
                ring_dimension=params['ring_dimension'],
                modulus_bits=params['modulus_bits'],
                field_bits=params['field_bits'],
                setup_rounds=params['setup_rounds'],
                online_rounds=params['online_rounds'],
                security_bits=params['security_bits'],
                security_margin=security_margin,
                protocol_type=params['protocol_type'],
                color=params['color'],
                marker=params['marker']
            )

        return protocols

    def get_protocol(self, name: str) -> Optional[ProtocolParameters]:
        """Get protocol parameters by name."""
        return self._protocols.get(name)

    def get_all_protocols(self) -> Dict[str, ProtocolParameters]:
        """Get all protocol configurations."""
        return self._protocols

    def get_protocol_names(self) -> list:
        """Get list of all protocol names."""
        return list(self._protocols.keys())

    def get_config_source(self) -> str:
        """Return configuration source file for traceability."""
        return f"config/{self._protocol_file}"

    @staticmethod
    def get_ot_baseline_protocols(baselines_file: str = "ot_baselines.yaml") -> Dict[str, dict]:
        """Get OT baseline protocol configurations for comparison."""
        config = load_yaml_config(baselines_file)
        common = config.get('common_parameters', {})
        field_bits = common.get('field_bits', 128)
        ec_point_bytes = common.get('ec_point_bytes', 32)

        baselines = {}

        # Simplest OT
        simplest = config.get('Simplest_OT', {})
        simplest_comm_per_ot = simplest.get('comm_per_ot_bytes', 64)
        simplest_comp_per_ot = simplest.get('comp_per_ot_aes_equiv', 4000)
        simplest_comm_per_ole = field_bits * simplest_comm_per_ot / 1024  # KB

        baselines['Simplest OT'] = {
            'name': 'Simplest OT',
            'comm_per_ole': simplest_comm_per_ole,
            'comp_per_ole': field_bits * simplest_comp_per_ot / 1e6,
            'rounds': simplest.get('rounds', 2),
            'type': 'OT',
            'color': simplest.get('color', '#E07A5F')
        }

        # IKNP OT Extension
        iknp = config.get('IKNP_OT', {})
        iknp_bits_per_ot = iknp.get('bits_per_extended_ot', 64)
        iknp_hash_ops = iknp.get('hash_ops_per_ot', 2)
        # Load AES_OPS_PER_HASH from constants
        constants = load_yaml_config('complexity_constants.yaml')
        aes_per_hash = constants.get('AES_OPS_PER_HASH', {}).get('value', 100)

        iknp_comm_per_ole = field_bits * iknp_bits_per_ot / 8 / 1024  # KB
        iknp_comp_per_ole = field_bits * iknp_hash_ops * aes_per_hash / 1e6

        baselines['IKNP OT'] = {
            'name': 'IKNP OT',
            'comm_per_ole': iknp_comm_per_ole,
            'comp_per_ole': iknp_comp_per_ole,
            'rounds': iknp.get('rounds', 4),
            'type': 'OT',
            'color': iknp.get('color', '#E07A5F')
        }

        # Silent OT
        silent = config.get('Silent_OT', {})
        seed_bits = silent.get('seed_bits', 128)
        seed_kb = seed_bits / 8 / 1024
        prg_ops = silent.get('prg_ops_per_output', 10)

        baselines['Silent OT'] = {
            'name': 'Silent OT',
            'comm_per_ole': seed_kb,
            'comp_per_ole': prg_ops / 1e6,
            'rounds': silent.get('rounds', 8),
            'scaling': silent.get('scaling', 'sublinear'),
            'type': 'OT',
            'color': silent.get('color', '#E07A5F')
        }

        return baselines
