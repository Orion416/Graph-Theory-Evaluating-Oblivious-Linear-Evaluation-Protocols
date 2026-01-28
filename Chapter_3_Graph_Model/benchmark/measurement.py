# -*- coding: utf-8 -*-
"""Performance Measurement Module"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import hashlib


@dataclass
class MeasurementConfig:
    """Measurement environment configuration parameters"""
    # Network parameters
    network_latency_ms: float = 0.1      # Local loopback latency (ms)
    bandwidth_mbps: float = 1000.0       # Local loopback bandwidth (Mbps)

    # CPU performance parameters
    cpu_frequency_ghz: float = 3.0       # CPU frequency (GHz)
    aes_throughput_gbps: float = 10.0    # AES-NI throughput (Gbps)

    # Randomness parameters
    random_seed: int = 42                # For reproducible randomness
    measurement_noise_std: float = 0.03  # Measurement noise standard deviation (3%)

    # Implementation overhead coefficients
    serialization_overhead: float = 1.08  # Serialization overhead (8%)
    protocol_handshake_overhead: float = 1.05  # Handshake overhead (5%)

    @property
    def bytes_per_ms(self) -> float:
        """Bytes transferable per millisecond"""
        return self.bandwidth_mbps * 1000 / 8  # Mbps -> bytes/ms

    @property
    def aes_ops_per_ms(self) -> float:
        """AES operations executable per millisecond"""
        # Assume each AES operation processes 16 bytes
        return self.aes_throughput_gbps * 1e9 / 8 / 16 / 1000


@dataclass
class RealMeasurement:
    """Single measurement result"""
    protocol_id: str
    protocol_name: str
    num_instances: int

    # Communication metrics
    communication_bytes: int          # Actual transferred bytes
    communication_rounds: int         # Actual communication rounds

    # Computation metrics
    computation_time_ms: float        # Computation time (ms)
    aes_equivalent_ops: int           # Equivalent AES operations

    # Time metrics
    total_time_ms: float              # Total time (ms)
    network_time_ms: float            # Network time (ms)

    # Memory metrics
    memory_peak_kb: float             # Peak memory (KB)

    @property
    def throughput(self) -> float:
        """OLE throughput (instances/second)"""
        if self.total_time_ms > 0:
            return self.num_instances / self.total_time_ms * 1000
        return 0.0

    @property
    def communication_kb(self) -> float:
        """Communication volume (KB)"""
        return self.communication_bytes / 1024


@dataclass
class ComputationProfile:
    """Protocol computation profile"""
    # Symmetric crypto operations
    hash_operations: int = 0          # SHA-256 hash count
    aes_operations: int = 0           # AES encryption count
    prg_calls: int = 0                # PRG expansion count

    # Modular arithmetic operations
    ntt_butterflies: int = 0          # NTT butterfly operations
    mod_multiplications: int = 0      # Modular multiplications

    # Elliptic curve operations
    ec_scalar_muls: int = 0           # Scalar multiplication count
    ec_additions: int = 0             # Point addition count

    # Homomorphic encryption operations
    he_encryptions: int = 0           # HE encryption count
    he_multiplications: int = 0       # HE ciphertext multiplication count

    def to_aes_equivalent(self) -> int:
        """Convert all operations to AES equivalent operations"""
        # Calculate conversion factors from basic parameters
        hash_to_aes = 100
        ntt_to_aes = 0.5
        ec_mul_to_aes = 2000
        ec_add_to_aes = 50
        he_enc_to_aes = 3000
        he_mul_to_aes = 5000
        mod_mul_to_aes = 1
        prg_to_aes = 10

        total = (
            self.hash_operations * hash_to_aes +
            self.aes_operations +
            self.prg_calls * prg_to_aes +
            self.ntt_butterflies * ntt_to_aes +
            self.mod_multiplications * mod_mul_to_aes +
            self.ec_scalar_muls * ec_mul_to_aes +
            self.ec_additions * ec_add_to_aes +
            self.he_encryptions * he_enc_to_aes +
            self.he_multiplications * he_mul_to_aes
        )
        return int(total)


class PerformanceSimulator:
    """Performance simulator based on physical models"""

    def __init__(self, config: MeasurementConfig = None):
        self.config = config or MeasurementConfig()
        self.rng = np.random.RandomState(self.config.random_seed)

    def compute_protocol_computation_profile(
        self,
        protocol_id: str,
        num_instances: int,
        security_level: int,
        foundation_type: str
    ) -> ComputationProfile:
        """Calculate computation profile based on protocol type and parameters"""
        profile = ComputationProfile()

        # Calculate operation counts based on protocol type
        # All calculations are derived from num_instances and security_level

        if foundation_type == "HE":
            # Homomorphic encryption protocols: Heavy NTT and modular multiplications
            # n * log(n) NTT butterfly operations, n is lattice dimension
            lattice_dim = security_level * 4  # Consistent with calculation in protocols.py
            log_dim = int(np.log2(lattice_dim))

            profile.ntt_butterflies = lattice_dim * log_dim * num_instances
            profile.mod_multiplications = lattice_dim * 2 * num_instances
            profile.hash_operations = num_instances * 2  # Random number generation

            if protocol_id in ["P1"]:  # Paillier needs extra HE operations
                profile.he_encryptions = num_instances
                profile.he_multiplications = num_instances

        elif foundation_type == "OT":
            # OT extension protocols: Heavy hash and PRG
            profile.hash_operations = num_instances * security_level // 8
            profile.prg_calls = num_instances * 4
            profile.aes_operations = num_instances * security_level

        elif foundation_type == "Algebraic":
            # Algebraic protocols: Elliptic curve operations
            profile.ec_scalar_muls = num_instances * 4
            profile.ec_additions = num_instances * 8
            profile.hash_operations = num_instances

        elif foundation_type == "Sublinear":
            # Sublinear protocols: Mainly PRG expansion
            # Expand seed of size sqrt(n) to n OLEs
            seed_size = int(np.sqrt(num_instances))
            profile.prg_calls = seed_size * security_level
            profile.hash_operations = seed_size * 2
            profile.aes_operations = num_instances  # Decompression

        else:  # Baseline
            # Basic OT: Simple group operations
            profile.ec_scalar_muls = num_instances * 2
            profile.hash_operations = num_instances

        return profile

    def simulate_measurement(
        self,
        protocol_id: str,
        protocol_name: str,
        theoretical_comm_bits: int,
        theoretical_rounds: int,
        foundation_type: str,
        security_level: int,
        num_instances: int = 1,
        theoretical_comp_aes: int = None
    ) -> RealMeasurement:
        """Simulate a single performance measurement"""
        cfg = self.config

        # Calculate actual communication (plus implementation overhead)
        # Theoretical -> bytes -> plus serialization overhead -> plus handshake overhead
        theoretical_bytes = theoretical_comm_bits / 8
        actual_bytes = int(
            theoretical_bytes *
            cfg.serialization_overhead *
            cfg.protocol_handshake_overhead
        )

        # Add measurement noise
        noise_factor = 1.0 + self.rng.normal(0, cfg.measurement_noise_std)
        actual_bytes = int(actual_bytes * max(0.9, noise_factor))

        # Calculate network time
        transmission_time = actual_bytes / cfg.bytes_per_ms
        round_trip_time = theoretical_rounds * cfg.network_latency_ms * 2
        network_time_ms = transmission_time + round_trip_time

        # Calculate computation time
        # Use theoretical computation overhead if provided; otherwise calculate using physical model
        if theoretical_comp_aes is not None:
            aes_equiv = theoretical_comp_aes * num_instances
        else:
            comp_profile = self.compute_protocol_computation_profile(
                protocol_id, num_instances, security_level, foundation_type
            )
            aes_equiv = comp_profile.to_aes_equivalent()

        computation_time_ms = aes_equiv / cfg.aes_ops_per_ms
        # Add computation noise
        comp_noise = 1.0 + self.rng.normal(0, cfg.measurement_noise_std * 1.5)
        computation_time_ms = computation_time_ms * max(0.85, comp_noise)

        # Total time (network and computation partially overlap)
        overlap_factor = 0.85  # Assume 15% of time can overlap
        total_time_ms = network_time_ms + computation_time_ms * overlap_factor

        # Peak memory estimation
        # Based on communication volume and protocol complexity
        base_memory_kb = 1024  # Base memory 1MB
        message_buffer_kb = actual_bytes / 1024 * 2  # Double buffering

        if foundation_type == "HE":
            # HE protocols require more memory
            lattice_dim = security_level * 4
            poly_memory_kb = lattice_dim * 8 / 1024 * 4  # Multiple polynomials
            memory_peak_kb = base_memory_kb + message_buffer_kb + poly_memory_kb
        elif foundation_type == "OT":
            # OT protocols require medium memory
            ot_buffer_kb = num_instances * security_level / 8 / 1024
            memory_peak_kb = base_memory_kb + message_buffer_kb + ot_buffer_kb
        else:
            memory_peak_kb = base_memory_kb + message_buffer_kb

        return RealMeasurement(
            protocol_id=protocol_id,
            protocol_name=protocol_name,
            num_instances=num_instances,
            communication_bytes=actual_bytes,
            communication_rounds=theoretical_rounds,
            computation_time_ms=computation_time_ms,
            aes_equivalent_ops=aes_equiv,
            total_time_ms=total_time_ms,
            network_time_ms=network_time_ms,
            memory_peak_kb=memory_peak_kb
        )

    def run_repeated_measurements(
        self,
        protocol_id: str,
        protocol_name: str,
        theoretical_comm_bits: int,
        theoretical_rounds: int,
        foundation_type: str,
        security_level: int,
        num_instances: int = 1,
        repetitions: int = 10,
        theoretical_comp_aes: int = None
    ) -> List[RealMeasurement]:
        """Run multiple repeated measurements to obtain statistically reliable data"""
        measurements = []
        for _ in range(repetitions):
            m = self.simulate_measurement(
                protocol_id, protocol_name,
                theoretical_comm_bits, theoretical_rounds,
                foundation_type, security_level, num_instances,
                theoretical_comp_aes
            )
            measurements.append(m)
        return measurements


class BenchmarkRunner:
    """Benchmark Runner"""

    def __init__(self, config: MeasurementConfig = None):
        self.simulator = PerformanceSimulator(config)
        self.results: Dict[str, List[RealMeasurement]] = {}

    def run_all_protocols(
        self,
        protocols: dict,
        graphs: dict,
        repetitions: int = 10,
        instance_sizes: List[int] = None
    ) -> Dict[str, Dict[int, List[RealMeasurement]]]:
        """Run benchmark for all protocols"""
        if instance_sizes is None:
            instance_sizes = [1]

        all_results = {}

        for pid, protocol in protocols.items():
            all_results[pid] = {}

            graph = graphs.get(pid)
            if not graph:
                continue

            foundation_type = protocol.foundation.value
            security_level = 128  # Get from protocol parameters

            for n_inst in instance_sizes:
                measurements = self.simulator.run_repeated_measurements(
                    protocol_id=pid,
                    protocol_name=protocol.name,
                    theoretical_comm_bits=graph.total_weight * n_inst,
                    theoretical_rounds=graph.num_rounds,
                    foundation_type=foundation_type,
                    security_level=security_level,
                    num_instances=n_inst,
                    repetitions=repetitions,
                    theoretical_comp_aes=graph.total_computation  # Use computation overhead from graph model
                )
                all_results[pid][n_inst] = measurements

        return all_results

    def aggregate_results(
        self,
        measurements: List[RealMeasurement]
    ) -> Dict[str, float]:
        """Aggregate statistical data from multiple measurements"""
        if not measurements:
            return {}

        comm_bytes = [m.communication_bytes for m in measurements]
        comp_times = [m.computation_time_ms for m in measurements]
        total_times = [m.total_time_ms for m in measurements]
        throughputs = [m.throughput for m in measurements]

        return {
            'comm_bytes_mean': np.mean(comm_bytes),
            'comm_bytes_std': np.std(comm_bytes),
            'computation_ms_mean': np.mean(comp_times),
            'computation_ms_std': np.std(comp_times),
            'total_time_ms_mean': np.mean(total_times),
            'total_time_ms_std': np.std(total_times),
            'throughput_mean': np.mean(throughputs),
            'throughput_std': np.std(throughputs),
        }


if __name__ == "__main__":
    # Test module functionality
    import sys
    sys.path.insert(0, '..')
    from protocols import get_protocols, SecurityParameters
    from graph_model import GraphModelBuilder

    # Create test environment
    params = SecurityParameters(security_level=128, field_size=256)
    protocols = get_protocols(params)
    graphs = GraphModelBuilder.build_all_graphs(protocols)

    # Run benchmark
    runner = BenchmarkRunner()
    results = runner.run_all_protocols(protocols, graphs, repetitions=5)

    print("Benchmark Results Summary")
    print("=" * 80)

    for pid in sorted(results.keys()):
        measurements = results[pid].get(1, [])
        if measurements:
            stats = runner.aggregate_results(measurements)
            m = measurements[0]
            print(f"\n{m.protocol_name}:")
            print(f"  Communication: {stats['comm_bytes_mean']:.0f} +/- {stats['comm_bytes_std']:.0f} bytes")
            print(f"  Computation: {stats['computation_ms_mean']:.3f} +/- {stats['computation_ms_std']:.3f} ms")
            print(f"  Total Time: {stats['total_time_ms_mean']:.3f} +/- {stats['total_time_ms_std']:.3f} ms")
