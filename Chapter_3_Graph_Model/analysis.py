# -*- coding: utf-8 -*-
"""Protocol Analysis and Comparison"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from protocols import Protocol, CryptoFoundation
from graph_model import CommunicationGraph
from metrics import GraphMetrics


@dataclass
class CategoryStatistics:
    """Statistics for a category of protocols."""
    category: str
    protocols: List[str]
    mean_edge_count: float
    mean_total_weight: float
    mean_balance: float
    mean_concentration: float


class ProtocolAnalyzer:
    """Analyzer for comparing protocols and extracting patterns."""

    @staticmethod
    def group_by_foundation(
        protocols: Dict[str, Protocol],
        metrics: Dict[str, GraphMetrics]
    ) -> Dict[str, CategoryStatistics]:
        """Group protocols by cryptographic foundation and compute category statistics."""
        # Group protocols by foundation
        foundation_groups = {}
        for pid, protocol in protocols.items():
            foundation = protocol.foundation.value
            if foundation not in foundation_groups:
                foundation_groups[foundation] = []
            foundation_groups[foundation].append(pid)

        # Compute statistics for each group
        category_stats = {}
        for foundation, pids in foundation_groups.items():
            group_metrics = [metrics[pid] for pid in pids]

            edge_counts = [m.num_edges for m in group_metrics]
            total_weights = [m.total_weight for m in group_metrics]
            balances = [m.edge_direction_balance for m in group_metrics]
            concentrations = [m.weight_concentration_index for m in group_metrics]

            category_stats[foundation] = CategoryStatistics(
                category=foundation,
                protocols=pids,
                mean_edge_count=np.mean(edge_counts),
                mean_total_weight=np.mean(total_weights),
                mean_balance=np.mean(balances),
                mean_concentration=np.mean(concentrations)
            )

        return category_stats

    @staticmethod
    def normalize_metrics(
        metrics: Dict[str, GraphMetrics]
    ) -> Dict[str, Dict[str, float]]:
        """Normalize metrics to [0, 1] range for comparison."""
        # Collect all values for each metric
        all_values = {
            'edge_count': [m.num_edges for m in metrics.values()],
            'total_weight': [m.total_weight for m in metrics.values()],
            'balance': [m.edge_direction_balance for m in metrics.values()],
            'concentration': [m.weight_concentration_index for m in metrics.values()],
            'entropy': [m.temporal_entropy for m in metrics.values()],
        }

        # Compute min and max
        min_vals = {k: min(v) for k, v in all_values.items()}
        max_vals = {k: max(v) for k, v in all_values.items()}

        # Normalize each protocol's metrics
        normalized = {}
        for pid, m in metrics.items():
            norm_dict = {}
            for metric_name, value in [
                ('edge_count', m.num_edges),
                ('total_weight', m.total_weight),
                ('balance', m.edge_direction_balance),
                ('concentration', m.weight_concentration_index),
                ('entropy', m.temporal_entropy),
            ]:
                range_val = max_vals[metric_name] - min_vals[metric_name]
                if range_val > 0:
                    norm_dict[metric_name] = (value - min_vals[metric_name]) / range_val
                else:
                    norm_dict[metric_name] = 0.5
            normalized[pid] = norm_dict

        return normalized

    @staticmethod
    def rank_protocols_by_weight(metrics: Dict[str, GraphMetrics]) -> List[str]:
        """Rank protocols by total edge weight in descending order."""
        sorted_pids = sorted(
            metrics.keys(),
            key=lambda x: metrics[x].total_weight,
            reverse=True
        )
        return sorted_pids

    @staticmethod
    def compute_spearman_rank_correlation(
        ranking1: List[str],
        ranking2: List[str]
    ) -> float:
        """Compute Spearman rank correlation between two protocol rankings."""
        # Get common protocols
        common = set(ranking1) & set(ranking2)
        if len(common) < 2:
            return 0.0

        # Get ranks for common protocols
        rank1 = {pid: i for i, pid in enumerate(ranking1) if pid in common}
        rank2 = {pid: i for i, pid in enumerate(ranking2) if pid in common}

        # Compute Spearman correlation
        n = len(common)
        d_squared = sum((rank1[pid] - rank2[pid])**2 for pid in common)

        rho = 1 - (6 * d_squared) / (n * (n**2 - 1))
        return rho


def print_category_table(category_stats: Dict[str, CategoryStatistics]) -> None:
    """Print category statistics table."""
    print("\nProtocol Characteristics by Cryptographic Foundation")
    print("=" * 100)
    print(f"{'Category':<20} {'Protocols':<15} {'Mean Edges':<12} {'Mean Weight':<12} "
          f"{'Mean Bal.':<12} {'Mean Conc.':<12}")
    print("-" * 100)

    # Define display order
    order = ['HE', 'OT', 'Algebraic', 'Sublinear', 'Baseline']

    for cat in order:
        if cat in category_stats:
            s = category_stats[cat]
            protocols_str = ', '.join(s.protocols)
            print(f"{s.category:<20} {protocols_str:<15} {s.mean_edge_count:<12.1f} "
                  f"{s.mean_total_weight:<12.0f} {s.mean_balance:<12.2f} "
                  f"{s.mean_concentration:<12.2f}")


if __name__ == "__main__":
    from protocols import get_protocols, SecurityParameters
    from graph_model import GraphModelBuilder
    from metrics import MetricsExtractor

    # Create protocols and compute metrics
    params = SecurityParameters(security_level=128, field_size=256)
    protocols = get_protocols(params)
    graphs = GraphModelBuilder.build_all_graphs(protocols)
    metrics = MetricsExtractor.extract_all_metrics(graphs)

    # Analyze by category
    analyzer = ProtocolAnalyzer()
    category_stats = analyzer.group_by_foundation(protocols, metrics)
    print_category_table(category_stats)

    # Rank protocols
    ranking = analyzer.rank_protocols_by_weight(metrics)
    print("\nProtocol Ranking by Total Weight (descending):")
    print("-" * 50)
    for i, pid in enumerate(ranking):
        m = metrics[pid]
        print(f"  {i+1}. {m.protocol_name}: {m.total_weight} bits")

    # Compute normalized metrics
    normalized = analyzer.normalize_metrics(metrics)
    print("\nNormalized Metrics for Radar Chart:")
    print("-" * 50)
    for pid in ['P1', 'P4', 'P8', 'P0']:
        print(f"\n{metrics[pid].protocol_name}:")
        for k, v in normalized[pid].items():
            print(f"  {k}: {v:.2f}")
