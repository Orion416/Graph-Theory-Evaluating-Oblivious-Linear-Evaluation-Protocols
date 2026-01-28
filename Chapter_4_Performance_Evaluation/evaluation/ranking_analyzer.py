# -*- coding: utf-8 -*-
"""
Ranking analyzer for OLE Protocol evaluation.

This module analyzes protocol rankings under different evaluation
methodologies for fairness validation.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulators.performance_simulator import PerformanceSimulator


@dataclass
class RankingComparison:
    """Container for ranking comparison results."""
    protocol: str
    framework_rank: int
    naive_comm_rank: int
    naive_comp_rank: int
    naive_combined_rank: int
    agreement: str  # 'Yes', 'Partial', 'No'


class RankingAnalyzer:
    """Analyzer for comparing rankings across evaluation methodologies."""

    def __init__(self, random_seed: int = 42):
        self.simulator = PerformanceSimulator(random_seed)

    def compute_naive_communication_ranking(self, scale: int = 16) -> Dict[str, int]:
        """Compute ranking based solely on communication complexity."""
        rankings = {}
        comm_data = {}

        for protocol in self.simulator.protocol_family.get_protocol_names():
            result = self.simulator.simulate_single_protocol(protocol, scale)
            comm_data[protocol] = result.communication_mean

        sorted_protocols = sorted(comm_data.items(), key=lambda x: x[1])
        for rank, (protocol, _) in enumerate(sorted_protocols, 1):
            rankings[protocol] = rank

        return rankings

    def compute_naive_computation_ranking(self, scale: int = 16) -> Dict[str, int]:
        """Compute ranking based solely on computational overhead."""
        rankings = {}
        comp_data = {}

        for protocol in self.simulator.protocol_family.get_protocol_names():
            result = self.simulator.simulate_single_protocol(protocol, scale)
            comp_data[protocol] = result.computation_mean

        sorted_protocols = sorted(comp_data.items(), key=lambda x: x[1])
        for rank, (protocol, _) in enumerate(sorted_protocols, 1):
            rankings[protocol] = rank

        return rankings

    def compute_naive_combined_ranking(self, scale: int = 16) -> Dict[str, int]:
        """Compute ranking based on simple sum of normalized metrics."""
        rankings = {}
        combined_data = {}

        # Get raw data
        protocols = self.simulator.protocol_family.get_protocol_names()
        comm_values = []
        comp_values = []

        for protocol in protocols:
            result = self.simulator.simulate_single_protocol(protocol, scale)
            comm_values.append(result.communication_mean)
            comp_values.append(result.computation_mean)

        # Simple min-max normalization (lower is better for both)
        comm_array = np.array(comm_values)
        comp_array = np.array(comp_values)

        comm_norm = (comm_array - comm_array.min()) / (comm_array.max() - comm_array.min())
        comp_norm = (comp_array - comp_array.min()) / (comp_array.max() - comp_array.min())

        # Combined score (lower is better)
        for i, protocol in enumerate(protocols):
            combined_data[protocol] = comm_norm[i] + comp_norm[i]

        sorted_protocols = sorted(combined_data.items(), key=lambda x: x[1])
        for rank, (protocol, _) in enumerate(sorted_protocols, 1):
            rankings[protocol] = rank

        return rankings

    def compute_framework_ranking(self, scale: int = 16) -> Dict[str, int]:
        """Compute ranking using the full evaluation framework."""
        from evaluation.performance_evaluator import PerformanceEvaluator

        evaluator = PerformanceEvaluator()
        results = evaluator.evaluate_at_scale(scale)

        return {protocol: result.rank for protocol, result in results.items()}

    def generate_table16_data(self, scale: int = 16) -> List[RankingComparison]:
        """Generate ranking comparison data for Table 16."""
        framework = self.compute_framework_ranking(scale)
        naive_comm = self.compute_naive_communication_ranking(scale)
        naive_comp = self.compute_naive_computation_ranking(scale)
        naive_combined = self.compute_naive_combined_ranking(scale)

        comparisons = []
        for protocol in self.simulator.protocol_family.get_protocol_names():
            fw_rank = framework[protocol]
            comm_rank = naive_comm[protocol]
            comp_rank = naive_comp[protocol]
            comb_rank = naive_combined[protocol]

            # Determine agreement level
            if fw_rank == comm_rank == comp_rank == comb_rank:
                agreement = 'Yes'
            elif fw_rank == comb_rank or abs(fw_rank - comb_rank) <= 1:
                agreement = 'Partial'
            else:
                agreement = 'No'

            comparisons.append(RankingComparison(
                protocol=protocol,
                framework_rank=fw_rank,
                naive_comm_rank=comm_rank,
                naive_comp_rank=comp_rank,
                naive_combined_rank=comb_rank,
                agreement=agreement
            ))

        # Sort by framework rank
        comparisons.sort(key=lambda x: x.framework_rank)

        return comparisons

    def generate_figure15_data(self, scale: int = 16) -> Dict[str, np.ndarray]:
        """Generate alluvial diagram data for Figure 15."""
        framework = self.compute_framework_ranking(scale)
        naive_comm = self.compute_naive_communication_ranking(scale)
        naive_comp = self.compute_naive_computation_ranking(scale)
        naive_combined = self.compute_naive_combined_ranking(scale)

        data = {}
        for protocol in self.simulator.protocol_family.get_protocol_names():
            data[protocol] = np.array([
                naive_comm[protocol],
                naive_comp[protocol],
                naive_combined[protocol],
                framework[protocol]
            ])

        return data

    def analyze_rank_reversals(self, scale: int = 16) -> Dict[str, List[Tuple[str, str]]]:
        """Identify rank reversals between evaluation methods."""
        rankings = {
            'naive_comm': self.compute_naive_communication_ranking(scale),
            'naive_comp': self.compute_naive_computation_ranking(scale),
            'naive_combined': self.compute_naive_combined_ranking(scale),
            'framework': self.compute_framework_ranking(scale)
        }

        protocols = self.simulator.protocol_family.get_protocol_names()
        reversals = {}

        for method in ['naive_comm', 'naive_comp', 'naive_combined']:
            reversals[f'{method}_vs_framework'] = []
            for i, p1 in enumerate(protocols):
                for p2 in protocols[i+1:]:
                    # Check if relative order changed
                    method_order = rankings[method][p1] < rankings[method][p2]
                    framework_order = rankings['framework'][p1] < rankings['framework'][p2]
                    if method_order != framework_order:
                        reversals[f'{method}_vs_framework'].append((p1, p2))

        return reversals
