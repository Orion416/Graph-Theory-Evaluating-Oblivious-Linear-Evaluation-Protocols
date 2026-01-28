# -*- coding: utf-8 -*-
"""Graph-Theoretic Modeling of Multiple OLE Protocols"""

import os
import sys
import numpy as np
from datetime import datetime

# Import project modules
from protocols import get_protocols, SecurityParameters, Protocol
from graph_model import GraphModelBuilder, CommunicationGraph, print_graph_summary
from metrics import MetricsExtractor, GraphMetrics, print_metrics_table, compute_metric_correlations
from analysis import ProtocolAnalyzer, print_category_table
from validation import (
    ConsistencyValidator, SelectionValidator,
    ExternalBenchmarkValidator,
    print_consistency_table, print_selection_results, print_benchmark_validation
)
from visualization import generate_all_figures

# Import benchmark modules
from benchmark import BenchmarkRunner, MeasurementConfig


def print_header():
    """Print analysis header."""
    print("=" * 80)
    print("Graph-Theoretic Modeling of Multiple OLE Protocols")
    print("Case Study Implementation")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_security_parameters(params: SecurityParameters):
    """Print the security parameters used."""
    print("\nSecurity Parameters")
    print("-" * 40)
    print(f"  Security Level: {params.security_level} bits")
    print(f"  Field Size: {params.field_size} bits")
    print(f"  Paillier Modulus: {params.paillier_modulus_bits} bits")
    print(f"  Group Element Size: {params.group_element_bits} bits")
    print(f"  Lattice Dimension: {params.lattice_dimension}")
    print(f"  Lattice Modulus: 2^{params.lattice_modulus_bits}")
    print()


def print_table2(graphs: dict):
    """Print Table 2: Communication Graph Model Parameters."""
    print("\n" + "=" * 90)
    print("Table 2: Communication Graph Model Parameters")
    print("=" * 90)
    print(f"{'Protocol':<15} {'|V|':<6} {'|E|':<6} {'Total Weight':<14} {'Rounds (r)':<11} {'Avg Weight':<12}")
    print("-" * 90)

    # Sort by protocol ID (P0 first as baseline, then P1-P8)
    sorted_keys = sorted(graphs.keys(), key=lambda x: (x != 'P0', x))

    for pid in sorted_keys:
        g = graphs[pid]
        avg_weight = g.total_weight / g.num_edges if g.num_edges > 0 else 0
        print(f"{g.protocol_name:<15} {g.num_vertices:<6} {g.num_edges:<6} "
              f"{g.total_weight:<14} {g.num_rounds:<11} {avg_weight:<12.0f}")


def print_table3(metrics: dict):
    """Print Table 3: Extracted Graph-Theoretic Metrics (Communication)."""
    print("\n" + "=" * 100)
    print("Table 3: Extracted Graph-Theoretic Metrics (Communication)")
    print("=" * 100)
    print(f"{'Protocol':<15} {'Diameter':<10} {'Max Degree':<12} {'Direction Bal.':<15} "
          f"{'Weight Conc.':<13} {'Temporal Ent.':<13}")
    print("-" * 100)

    sorted_keys = sorted(metrics.keys(), key=lambda x: (x != 'P0', x))

    for pid in sorted_keys:
        m = metrics[pid]
        print(f"{m.protocol_name:<15} {m.graph_diameter:<10} {m.max_vertex_degree:<12} "
              f"{m.edge_direction_balance:<15.2f} {m.weight_concentration_index:<13.2f} "
              f"{m.temporal_entropy:<13.2f}")


def print_table3_5(metrics: dict):
    """Print Table 3.5: Computation Metrics."""
    print("\n" + "=" * 110)
    print("Table 3.5: Computation Complexity Metrics")
    print("=" * 110)
    print(f"{'Protocol':<15} {'Total Comp.':<14} {'Comp. Bal.':<12} {'Comp. Conc.':<12} "
          f"{'Comm/Comp':<12} {'Category':<15}")
    print("-" * 110)

    sorted_keys = sorted(metrics.keys(), key=lambda x: (x != 'P0', x))

    for pid in sorted_keys:
        m = metrics[pid]
        # Determine category based on ratio
        if m.comm_to_comp_ratio == float('inf'):
            category = "Comm-dominant"
            ratio_str = "∞"
        elif m.comm_to_comp_ratio > 1.0:
            category = "Comm-intensive"
            ratio_str = f"{m.comm_to_comp_ratio:.4f}"
        elif m.comm_to_comp_ratio > 0.1:
            category = "Balanced"
            ratio_str = f"{m.comm_to_comp_ratio:.4f}"
        else:
            category = "Comp-intensive"
            ratio_str = f"{m.comm_to_comp_ratio:.4f}"

        print(f"{m.protocol_name:<15} {m.total_computation:<14} {m.computation_balance:<12.2f} "
              f"{m.computation_concentration:<12.2f} {ratio_str:<12} {category:<15}")


def print_table4(protocols: dict, metrics: dict):
    """Print Table 4: Protocol Characteristics by Cryptographic Foundation."""
    analyzer = ProtocolAnalyzer()
    category_stats = analyzer.group_by_foundation(protocols, metrics)

    print("\n" + "=" * 100)
    print("Table 4: Protocol Characteristics Grouped by Cryptographic Foundation")
    print("=" * 100)
    print(f"{'Category':<22} {'Protocols':<15} {'Mean Edges':<12} {'Mean Weight':<13} "
          f"{'Mean Balance':<13} {'Mean Conc.':<12}")
    print("-" * 100)

    # Define display order
    order = ['HE', 'OT', 'Algebraic', 'Sublinear', 'Baseline']

    for cat in order:
        if cat in category_stats:
            s = category_stats[cat]
            protocols_str = ', '.join(s.protocols)
            print(f"{s.category:<22} {protocols_str:<15} {s.mean_edge_count:<12.1f} "
                  f"{s.mean_total_weight:<13.0f} {s.mean_balance:<13.2f} "
                  f"{s.mean_concentration:<12.2f}")


def compute_method_scores():
    """Compute comparison scores for protocol description methods."""
    # Define capabilities (boolean)
    # [NL, PC, AN, GM]
    has_formal_syntax = np.array([0, 1, 1, 1])
    has_precise_values = np.array([0, 0, 0, 1])
    has_structure_repr = np.array([0, 1, 0, 1])
    has_temporal_order = np.array([0, 1, 0, 1])
    has_quantitative = np.array([0, 0, 0, 1])
    has_algorithmic = np.array([0, 0, 0, 1])
    is_human_readable = np.array([1, 1, 0, 0])
    is_standardized = np.array([0, 0, 1, 0])

    # Aggregate scores (normalized 0-1)
    expressiveness = has_formal_syntax * 0.4 + has_structure_repr * 0.6
    precision = has_precise_values * 0.5 + has_structure_repr * 0.5
    temporal = has_temporal_order
    comparison = has_quantitative
    algorithmic = has_algorithmic
    readability = is_human_readable * 0.7 + has_formal_syntax * 0.3
    standard = is_standardized

    return {
        'Expressiveness for communication': expressiveness.astype(float),
        'Precision of message sizes': precision.astype(float),
        'Preservation of temporal ordering': temporal.astype(float),
        'Support for quantitative comparison': comparison.astype(float),
        'Amenability to algorithmic analysis': algorithmic.astype(float),
        'Human readability': readability.astype(float),
        'Standardization across publications': standard.astype(float),
    }


def score_to_label(score):
    """Convert numeric score to qualitative label."""
    if score >= 0.9:
        return 'High'
    elif score >= 0.6:
        return 'Moderate'
    elif score >= 0.3:
        return 'Low'
    elif score > 0:
        return 'Low'
    else:
        return 'None'


def print_table5():
    """Print Table 5: Comparison of Protocol Description Methods."""
    print("\n" + "=" * 100)
    print("Table 5: Comparison of Protocol Description Methods")
    print("=" * 100)

    methods = ['Natural Language', 'Pseudocode', 'Asymptotic', 'Graph Model']

    # Compute scores
    dimension_scores = compute_method_scores()

    print(f"{'Evaluation Dimension':<35} ", end='')
    for method in methods:
        print(f"{method:<15} ", end='')
    print()
    print("-" * 100)

    for dim_name, scores in dimension_scores.items():
        print(f"{dim_name:<35} ", end='')
        for i, score in enumerate(scores):
            # Special case for Graph Model standardization
            if dim_name == 'Standardization across publications' and i == 3:
                label = 'Proposed'
            else:
                label = score_to_label(score)
            print(f"{label:<15} ", end='')
        print()


def print_table6(graphs: dict, params: SecurityParameters):
    """Print Table 6: Consistency Validation."""
    validations = ConsistencyValidator.validate_all(graphs, params)

    print("\n" + "=" * 110)
    print("Table 6: Consistency Validation of Graph Model Total Edge Weights")
    print("=" * 110)
    print(f"{'Protocol':<15} {'Model Weight':<14} {'Reference Specification':<38} {'Rel. Diff.':<12}")
    print("-" * 110)

    sorted_keys = sorted(validations.keys(), key=lambda x: (x != 'P0', x))

    for pid in sorted_keys:
        v = validations[pid]
        print(f"{v.protocol_name:<15} {v.graph_model_weight:<14} "
              f"{v.reference_specification:<38} {v.relative_difference:<12.2f}%")


def print_table7(benchmark_summary):
    """Print Table 7: External Benchmark Validation."""
    if benchmark_summary is None:
        print("\n[Table 7 skipped: No benchmark data available]")
        return

    print("\n" + "=" * 130)
    print("Table 7: External Benchmark Validation")
    print("=" * 130)
    print(f"{'Protocol':<15} {'Model (bits)':<14} {'Measured (B)':<14} {'Comm. Err':<12} "
          f"{'Model (AES)':<14} {'Measured (ms)':<14} {'Rank Score':<12}")
    print("-" * 130)

    # Sort by protocol ID
    sorted_results = sorted(benchmark_summary.protocol_results,
                           key=lambda x: (x.protocol_id != 'P0', x.protocol_id))

    for r in sorted_results:
        # Format output
        model_comm = f"{r.model_comm_bits:,}"
        measured_comm = f"{r.measured_comm_bytes:,.0f}"
        comm_err = f"{r.comm_relative_error:.2%}"
        model_comp = f"{r.model_comp_aes:,}"
        measured_comp = f"{r.measured_comp_ms:.2f}"
        comp_score = f"{r.comp_relative_score:.2f}"

        print(f"{r.protocol_name:<15} {model_comm:<14} {measured_comm:<14} {comm_err:<12} "
              f"{model_comp:<14} {measured_comp:<14} {comp_score:<12}")

    # Summary stats
    print("-" * 130)
    print(f"Spearman Correlation: Comm. ρ = {benchmark_summary.comm_spearman_rho:.4f}, "
          f"Comp. ρ = {benchmark_summary.comp_spearman_rho:.4f}")
    print(f"Kendall τ = {benchmark_summary.kendall_tau:.4f}, "
          f"Mean Comm. Error = {benchmark_summary.comm_mean_error:.2%}")
    print(f"Validation: {benchmark_summary.validity_message}")


def save_tables_to_file(
    graphs: dict,
    metrics: dict,
    protocols: dict,
    params: SecurityParameters,
    output_dir: str,
    benchmark_summary=None
):
    """Save all tables to a text file."""
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    filepath = os.path.join(tables_dir, 'all_tables.txt')

    # Redirect stdout to file
    original_stdout = sys.stdout
    with open(filepath, 'w', encoding='utf-8') as f:
        sys.stdout = f

        print_header()
        print_security_parameters(params)
        print_table2(graphs)
        print_table3(metrics)
        print_table3_5(metrics)  # Computation metrics
        print_table4(protocols, metrics)
        print_table5()
        print_table6(graphs, params)
        print_table7(benchmark_summary)  # Benchmark validation

        # Print selection results
        selection_results = SelectionValidator.validate_selection_accuracy(metrics, protocols)
        print_selection_results(selection_results)

        # Print benchmark details
        if benchmark_summary:
            print_benchmark_validation(benchmark_summary)

        # Print correlations
        correlations = compute_metric_correlations(metrics)
        print("\n" + "=" * 60)
        print("Metric Correlations")
        print("=" * 60)
        for name, value in correlations.items():
            print(f"  {name}: {value:.2f}")

        sys.stdout = original_stdout

    print(f"\nTables saved to: {filepath}")


def run_analysis():
    """Run the complete analysis pipeline."""
    print_header()

    # Step 1: Initialize parameters
    print("Step 1: Initializing security parameters...")
    params = SecurityParameters(security_level=128, field_size=256)
    print_security_parameters(params)

    # Step 2: Create protocols
    print("Step 2: Creating protocol definitions...")
    protocols = get_protocols(params)
    print(f"  Created {len(protocols)} protocol definitions")

    # Step 3: Build graphs
    print("\nStep 3: Building communication graphs...")
    graphs = GraphModelBuilder.build_all_graphs(protocols)
    print(f"  Built {len(graphs)} communication graphs")

    # Step 4: Extract metrics
    print("\nStep 4: Extracting graph-theoretic metrics...")
    metrics = MetricsExtractor.extract_all_metrics(graphs)
    print(f"  Extracted metrics for {len(metrics)} protocols")

    # Step 5: Run benchmark (simulation)
    print("\nStep 5: Running benchmark tests...")
    benchmark_config = MeasurementConfig(
        network_latency_ms=0.1,       # 0.1ms LAN latency
        bandwidth_mbps=1000.0,        # 1 Gbps bandwidth
        cpu_frequency_ghz=3.0,        # 3.0 GHz CPU
        measurement_noise_std=0.03,   # 3% noise
        serialization_overhead=1.08   # 8% overhead
    )
    benchmark_runner = BenchmarkRunner(config=benchmark_config)
    benchmark_results = benchmark_runner.run_all_protocols(
        protocols=protocols,
        graphs=graphs,
        repetitions=10,
        instance_sizes=[1]
    )
    print(f"  Completed benchmark for {len(benchmark_results)} protocols")

    # Step 6: External validation
    print("\nStep 6: External benchmark validation...")
    benchmark_validator = ExternalBenchmarkValidator(graphs, metrics)
    benchmark_summary = benchmark_validator.validate_with_benchmark(benchmark_results)
    print(f"  Validation complete: Spearman ρ = {benchmark_summary.comm_spearman_rho:.4f}")

    # Step 7: Output tables
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print_table2(graphs)
    print_table3(metrics)
    print_table3_5(metrics)
    print_table4(protocols, metrics)
    print_table5()
    print_table6(graphs, params)
    print_table7(benchmark_summary)

    # Step 8: Validation analysis
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    selection_results = SelectionValidator.validate_selection_accuracy(metrics, protocols)
    print_selection_results(selection_results)

    # Print benchmark details
    print_benchmark_validation(benchmark_summary)

    # Print correlations
    correlations = compute_metric_correlations(metrics)
    print("\nMetric Correlations:")
    print("-" * 50)
    for name, value in correlations.items():
        print(f"  {name}: {value:.2f}")

    # Step 9: Generate figures
    print("\n" + "=" * 80)
    print("FIGURE GENERATION")
    print("=" * 80)

    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # Pass benchmark_summary for Figure 7
    generate_all_figures(protocols, graphs, metrics, selection_results, output_dir,
                        benchmark_summary=benchmark_summary)

    # Step 10: Save tables
    output_base = os.path.join(os.path.dirname(__file__), 'output')
    save_tables_to_file(graphs, metrics, protocols, params, output_base,
                       benchmark_summary=benchmark_summary)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_base}")
    print("  - figures/svg/: SVG format figures")
    print("  - figures/png/: PNG format figures")
    print("  - tables/: Text file with all tables")


if __name__ == "__main__":
    run_analysis()
