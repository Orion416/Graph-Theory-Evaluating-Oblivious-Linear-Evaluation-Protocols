# -*- coding: utf-8 -*-
"""Communication Graph Model Construction"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from protocols import Protocol, Message


@dataclass
class WeightedEdge:
    """Weighted edge with communication and computation weights."""
    source: int
    target: int
    round_num: int
    comm_weight: int
    comp_weight: int


@dataclass
class CommunicationGraph:
    """Represents a communication graph for a protocol."""
    protocol_id: str
    protocol_name: str
    num_vertices: int
    edges: List[Tuple[int, int, int]]  # (source, target, weight) - for backward compatibility
    temporal_sequence: List[List[Tuple[int, int, int]]] = field(default_factory=list)

    # Extended attribute: weighted edges
    weighted_edges: List[WeightedEdge] = field(default_factory=list)

    @property
    def num_edges(self) -> int:
        """Total number of edges in the graph."""
        return len(self.edges)

    @property
    def total_weight(self) -> int:
        """Sum of all edge weights."""
        return sum(w for _, _, w in self.edges)

    @property
    def num_rounds(self) -> int:
        """Number of temporal rounds."""
        return len(self.temporal_sequence)

    def get_adjacency_matrix(self) -> np.ndarray:
        """Compute the weighted adjacency matrix."""
        adj = np.zeros((self.num_vertices, self.num_vertices))
        for src, tgt, weight in self.edges:
            adj[src, tgt] += weight
        return adj

    def get_edge_count_matrix(self) -> np.ndarray:
        """Compute the edge count matrix."""
        count = np.zeros((self.num_vertices, self.num_vertices), dtype=int)
        for src, tgt, _ in self.edges:
            count[src, tgt] += 1
        return count

    def get_in_degrees(self) -> np.ndarray:
        """Compute in-degree for each vertex."""
        count_matrix = self.get_edge_count_matrix()
        return count_matrix.sum(axis=0)

    def get_out_degrees(self) -> np.ndarray:
        """Compute out-degree for each vertex."""
        count_matrix = self.get_edge_count_matrix()
        return count_matrix.sum(axis=1)

    def get_total_degrees(self) -> np.ndarray:
        """Compute total degree (in + out) for each vertex."""
        return self.get_in_degrees() + self.get_out_degrees()

    def get_weights_by_direction(self) -> Tuple[int, int]:
        """Compute total weights by direction."""
        w_0_to_1 = sum(w for s, t, w in self.edges if s == 0 and t == 1)
        w_1_to_0 = sum(w for s, t, w in self.edges if s == 1 and t == 0)
        return w_0_to_1, w_1_to_0

    def get_round_weights(self) -> List[int]:
        """Compute total weight for each round."""
        return [sum(w for _, _, w in round_edges)
                for round_edges in self.temporal_sequence]

    @property
    def total_computation(self) -> int:
        """Total computation cost (AES ops)."""
        if self.weighted_edges:
            return sum(e.comp_weight for e in self.weighted_edges)
        return 0

    def get_computation_by_direction(self) -> Tuple[int, int]:
        """Get computation cost by direction."""
        comp_0_to_1 = sum(e.comp_weight for e in self.weighted_edges
                         if e.source == 0 and e.target == 1)
        comp_1_to_0 = sum(e.comp_weight for e in self.weighted_edges
                         if e.source == 1 and e.target == 0)
        return comp_0_to_1, comp_1_to_0

    def get_computation_balance(self) -> float:
        """Computation balance."""
        comp_0_to_1, comp_1_to_0 = self.get_computation_by_direction()
        if comp_1_to_0 == 0:
            return float('inf') if comp_0_to_1 > 0 else 1.0
        return comp_0_to_1 / comp_1_to_0

    def get_round_computations(self) -> List[int]:
        """Get computation cost per round."""
        if not self.weighted_edges:
            return []

        max_round = max(e.round_num for e in self.weighted_edges)
        round_comps = []

        for r in range(1, max_round + 1):
            round_comp = sum(e.comp_weight for e in self.weighted_edges
                           if e.round_num == r)
            round_comps.append(round_comp)

        return round_comps

    @property
    def comm_to_comp_ratio(self) -> float:
        """Communication to computation ratio."""
        if self.total_computation == 0:
            return float('inf')
        # Convert communication to bytes for comparison
        comm_bytes = self.total_weight / 8
        return comm_bytes / self.total_computation


class GraphModelBuilder:
    """Builder class for constructing communication graphs from protocols."""

    @staticmethod
    def build_from_protocol(protocol: Protocol) -> CommunicationGraph:
        """Build a communication graph from a protocol definition."""
        # Extract edges from messages (backward compatibility)
        edges = [(msg.sender, msg.receiver, msg.size_bits)
                 for msg in protocol.messages]

        # Build temporal sequence
        max_round = protocol.num_rounds
        temporal_sequence = []

        for r in range(1, max_round + 1):
            round_edges = [(msg.sender, msg.receiver, msg.size_bits)
                          for msg in protocol.messages if msg.round_num == r]
            temporal_sequence.append(round_edges)

        # Build weighted edges
        weighted_edges = []
        for msg in protocol.messages:
            # Get computation cost from message
            comp_cost = msg.computation.to_aes_equivalent() if hasattr(msg, 'computation') else 0

            weighted_edge = WeightedEdge(
                source=msg.sender,
                target=msg.receiver,
                round_num=msg.round_num,
                comm_weight=msg.size_bits,
                comp_weight=comp_cost
            )
            weighted_edges.append(weighted_edge)

        return CommunicationGraph(
            protocol_id=protocol.identifier,
            protocol_name=protocol.name,
            num_vertices=protocol.num_parties,
            edges=edges,
            temporal_sequence=temporal_sequence,
            weighted_edges=weighted_edges
        )

    @staticmethod
    def build_all_graphs(protocols: Dict[str, Protocol]) -> Dict[str, CommunicationGraph]:
        """Build communication graphs for all protocols."""
        graphs = {}
        for pid, protocol in protocols.items():
            graphs[pid] = GraphModelBuilder.build_from_protocol(protocol)
        return graphs


def print_graph_summary(graphs: Dict[str, CommunicationGraph]) -> None:
    """Print summary of all communication graphs."""
    print("\nCommunication Graph Model Summary")
    print("=" * 80)
    print(f"{'Protocol':<15} {'|V|':<5} {'|E|':<5} {'Total Weight':<14} {'Rounds':<7} {'Avg Weight':<12}")
    print("-" * 80)

    for pid in sorted(graphs.keys(), key=lambda x: (x != 'P0', x)):
        g = graphs[pid]
        avg_weight = g.total_weight / g.num_edges if g.num_edges > 0 else 0
        print(f"{g.protocol_name:<15} {g.num_vertices:<5} {g.num_edges:<5} "
              f"{g.total_weight:<14} {g.num_rounds:<7} {avg_weight:<12.0f}")


if __name__ == "__main__":
    from protocols import get_protocols, SecurityParameters

    # Create protocols with standard parameters
    params = SecurityParameters(security_level=128, field_size=256)
    protocols = get_protocols(params)

    # Build graphs
    graphs = GraphModelBuilder.build_all_graphs(protocols)

    # Print summary
    print_graph_summary(graphs)

    # Print adjacency matrices
    print("\n" + "=" * 80)
    print("Adjacency Matrices (weighted)")
    print("=" * 80)

    for pid in ["P1", "P3", "P4", "P8"]:
        g = graphs[pid]
        adj = g.get_adjacency_matrix()
        print(f"\n{g.protocol_name}:")
        print(f"  v1->v2: {adj[0,1]:.0f} bits")
        print(f"  v2->v1: {adj[1,0]:.0f} bits")
