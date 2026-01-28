# -*- coding: utf-8 -*-
"""Graph-Theoretic Metrics Extraction"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from graph_model import CommunicationGraph


@dataclass
class GraphMetrics:
    """Container for all computed metrics of a communication graph."""
    protocol_id: str
    protocol_name: str

    # Topological metrics
    graph_diameter: int
    max_vertex_degree: int

    # Communication weighted metrics
    edge_direction_balance: float
    weight_concentration_index: float

    # Temporal metrics
    temporal_entropy: float

    # Computation metrics
    total_computation: int              # Total computation cost (AES ops)
    computation_balance: float          # Computation balance
    computation_concentration: float    # Computation concentration
    comm_to_comp_ratio: float           # Comm/Comp ratio

    # Basic statistics
    num_vertices: int
    num_edges: int
    total_weight: int
    num_rounds: int
    avg_edge_weight: float


class MetricsExtractor:
    """Extracts graph-theoretic metrics from communication graphs."""

    @staticmethod
    def compute_graph_diameter(graph: CommunicationGraph) -> int:
        """Compute the diameter of the communication graph."""
        n = graph.num_vertices
        if n <= 1:
            return 0

        # Build connectivity matrix
        adj = graph.get_edge_count_matrix()

        # For directed graphs, consider both directions
        connected = (adj + adj.T) > 0

        # Floyd-Warshall for shortest paths
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)

        for i in range(n):
            for j in range(n):
                if connected[i, j]:
                    dist[i, j] = 1

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        # Find maximum finite distance
        finite_dist = dist[dist < np.inf]
        if len(finite_dist) == 0:
            return 0

        diameter = int(np.max(finite_dist))
        return diameter

    @staticmethod
    def compute_max_vertex_degree(graph: CommunicationGraph) -> int:
        """Compute the maximum vertex degree."""
        in_degrees = graph.get_in_degrees()
        out_degrees = graph.get_out_degrees()
        max_in = int(np.max(in_degrees)) if len(in_degrees) > 0 else 0
        max_out = int(np.max(out_degrees)) if len(out_degrees) > 0 else 0
        return max(max_in, max_out)

    @staticmethod
    def compute_edge_direction_balance(graph: CommunicationGraph) -> float:
        """Compute the edge direction balance metric."""
        w_0_to_1, w_1_to_0 = graph.get_weights_by_direction()

        if w_1_to_0 == 0:
            return float('inf') if w_0_to_1 > 0 else 1.0

        return w_0_to_1 / w_1_to_0

    @staticmethod
    def compute_weight_concentration_index(graph: CommunicationGraph) -> float:
        """Compute the weight concentration index using normalized Gini coefficient."""
        weights = np.array([w for _, _, w in graph.edges])

        if len(weights) == 0:
            return 0.0

        if len(weights) == 1:
            return 0.0

        # Sort weights
        weights = np.sort(weights)
        n = len(weights)

        # Compute Gini coefficient
        cumulative = np.cumsum(weights)
        total = cumulative[-1]

        if total == 0:
            return 0.0

        # Gini = 1 - 2 * (area under Lorenz curve)
        # Area under Lorenz curve = sum(cumulative) / (n * total) - 0.5 / n
        lorenz_area = np.sum(cumulative) / (n * total)
        gini = 1 - 2 * lorenz_area + 1 / n

        # Normalize to [0, 1]
        gini = max(0.0, min(1.0, gini))

        return gini

    @staticmethod
    def compute_temporal_entropy(graph: CommunicationGraph) -> float:
        """Compute the temporal entropy of communication distribution."""
        round_weights = graph.get_round_weights()

        if not round_weights or sum(round_weights) == 0:
            return 0.0

        # Convert to probability distribution
        total = sum(round_weights)
        probs = np.array([w / total for w in round_weights if w > 0])

        if len(probs) == 0:
            return 0.0

        # Shannon entropy: H = -sum(p * log2(p))
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return entropy

    @staticmethod
    def compute_computation_balance(graph: CommunicationGraph) -> float:
        """Compute computation balance."""
        return graph.get_computation_balance()

    @staticmethod
    def compute_computation_concentration(graph: CommunicationGraph) -> float:
        """Compute computation concentration (normalized Gini coefficient)."""
        if not graph.weighted_edges:
            return 0.0

        comp_weights = np.array([e.comp_weight for e in graph.weighted_edges])

        if len(comp_weights) == 0:
            return 0.0

        if len(comp_weights) == 1:
            return 0.0

        # Sort computation weights
        comp_weights = np.sort(comp_weights)
        n = len(comp_weights)

        # Compute Gini coefficient
        cumulative = np.cumsum(comp_weights)
        total = cumulative[-1]

        if total == 0:
            return 0.0

        lorenz_area = np.sum(cumulative) / (n * total)
        gini = 1 - 2 * lorenz_area + 1 / n

        return max(0.0, min(1.0, gini))

    @classmethod
    def extract_metrics(cls, graph: CommunicationGraph) -> GraphMetrics:
        """Extract all metrics from a communication graph."""
        return GraphMetrics(
            protocol_id=graph.protocol_id,
            protocol_name=graph.protocol_name,
            # Topological metrics
            graph_diameter=cls.compute_graph_diameter(graph),
            max_vertex_degree=cls.compute_max_vertex_degree(graph),
            # Communication metrics
            edge_direction_balance=cls.compute_edge_direction_balance(graph),
            weight_concentration_index=cls.compute_weight_concentration_index(graph),
            # Temporal metrics
            temporal_entropy=cls.compute_temporal_entropy(graph),
            # Computation metrics
            total_computation=graph.total_computation,
            computation_balance=cls.compute_computation_balance(graph),
            computation_concentration=cls.compute_computation_concentration(graph),
            comm_to_comp_ratio=graph.comm_to_comp_ratio,
            # Basic statistics
            num_vertices=graph.num_vertices,
            num_edges=graph.num_edges,
            total_weight=graph.total_weight,
            num_rounds=graph.num_rounds,
            avg_edge_weight=graph.total_weight / graph.num_edges if graph.num_edges > 0 else 0
        )

    @classmethod
    def extract_all_metrics(cls, graphs: Dict[str, CommunicationGraph]) -> Dict[str, GraphMetrics]:
        """Extract metrics from all graphs."""
        metrics = {}
        for pid, graph in graphs.items():
            metrics[pid] = cls.extract_metrics(graph)
        return metrics


def compute_metric_correlations(metrics: Dict[str, GraphMetrics]) -> Dict[str, float]:
    """Compute Pearson correlation coefficients between metric pairs."""
    # Extract metric arrays
    edge_counts = np.array([m.num_edges for m in metrics.values()])
    total_weights = np.array([m.total_weight for m in metrics.values()])
    temporal_entropies = np.array([m.temporal_entropy for m in metrics.values()])
    concentrations = np.array([m.weight_concentration_index for m in metrics.values()])

    # Computation metric arrays
    total_computations = np.array([m.total_computation for m in metrics.values()])
    comp_concentrations = np.array([m.computation_concentration for m in metrics.values()])

    def pearson_corr(x, y):
        """Compute Pearson correlation coefficient."""
        if len(x) < 2:
            return 0.0
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        if denominator == 0:
            return 0.0
        return np.sum(x_centered * y_centered) / denominator

    correlations = {
        # Communication metric correlations
        'edge_count_vs_temporal_entropy': pearson_corr(edge_counts, temporal_entropies),
        'total_weight_vs_concentration': pearson_corr(total_weights, concentrations),
        # Comm-Comp cross correlations
        'total_weight_vs_total_computation': pearson_corr(total_weights, total_computations),
        'weight_conc_vs_comp_conc': pearson_corr(concentrations, comp_concentrations),
        # Computation metric correlations
        'edge_count_vs_computation': pearson_corr(edge_counts, total_computations),
    }

    return correlations


def print_metrics_table(metrics: Dict[str, GraphMetrics]) -> None:
    """Print metrics in table format."""
    print("\nExtracted Graph-Theoretic Metrics")
    print("=" * 100)
    print(f"{'Protocol':<15} {'Diameter':<9} {'Max Deg':<9} {'Balance':<9} "
          f"{'Conc.':<9} {'Entropy':<9}")
    print("-" * 100)

    for pid in sorted(metrics.keys(), key=lambda x: (x != 'P0', x)):
        m = metrics[pid]
        print(f"{m.protocol_name:<15} {m.graph_diameter:<9} {m.max_vertex_degree:<9} "
              f"{m.edge_direction_balance:<9.2f} {m.weight_concentration_index:<9.2f} "
              f"{m.temporal_entropy:<9.2f}")


if __name__ == "__main__":
    from protocols import get_protocols, SecurityParameters
    from graph_model import GraphModelBuilder

    # Create protocols and graphs
    params = SecurityParameters(security_level=128, field_size=256)
    protocols = get_protocols(params)
    graphs = GraphModelBuilder.build_all_graphs(protocols)

    # Extract metrics
    metrics = MetricsExtractor.extract_all_metrics(graphs)

    # Print results
    print_metrics_table(metrics)

    # Compute correlations
    correlations = compute_metric_correlations(metrics)
    print("\nMetric Correlations:")
    print("-" * 50)
    for name, value in correlations.items():
        print(f"  {name}: {value:.2f}")
