# -*- coding: utf-8 -*-
"""
Main entry point for Chapter 4 Evaluation Framework.
Generates all tables and figures described in the thesis.
"""

import os
import sys
from visualization.figure_generator import FigureGenerator
from evaluation.performance_evaluator import PerformanceEvaluator, CorrelationAnalyzer
from simulators.performance_simulator import PerformanceSimulator

def print_header(title):
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

def generate_tables():
    print_header("Generating Tables for Chapter 4")
    
    # Table 14
    print("\nTable 14: Complete Protocol Performance Ranking with Score Decomposition")
    print("-" * 100)
    print(f"{'Rank':<5} {'Protocol':<10} {'Overall':<10} {'Comm.':<10} {'Round':<10} {'Comp.':<10} {'Security':<10}")
    print("-" * 100)
    
    evaluator = PerformanceEvaluator()
    data = evaluator.generate_table14_data(scale=16)
    
    for row in data:
        print(f"{row['rank']:<5} {row['protocol']:<10} {row['overall_score']:<10.3f} "
              f"{row['comm_score']:<10.3f} {row['round_score']:<10.3f} "
              f"{row['comp_score']:<10.3f} {row['security_score']:<10.3f}")

    # Table 15
    print("\n\nTable 15: OLE Protocol Comparison with OT Baselines")
    print("-" * 100)
    print(f"{'Protocol':<15} {'Comm (KB)':<15} {'Rounds':<10} {'Comp (M ops)':<15} {'Type':<10}")
    print("-" * 100)
    
    simulator = PerformanceSimulator()
    data_t15 = simulator.generate_table15_data(scale=16)
    sorted_keys = sorted(data_t15.keys(), key=lambda x: (data_t15[x]['type'] == 'OT', x))
    
    for name in sorted_keys:
        row = data_t15[name]
        print(f"{name:<15} {row['communication']:<15.1f} {row['rounds']:<10} "
              f"{row['computation']:<15.2f} {row['type']:<10}")

    # Table 17
    print("\n\nTable 17: Application-Level Validation")
    print("-" * 100)
    print(f"{'Protocol':<10} {'F-Rank':<10} {'Obs-Rank':<10} {'Exec Time (s)':<15} {'Match':<10}")
    print("-" * 100)
    
    analyzer = CorrelationAnalyzer(evaluator, simulator)
    data_t17 = analyzer.generate_table17_data(scale=16)
    sorted_keys_t17 = sorted(data_t17.keys(), key=lambda x: data_t17[x]['framework_rank'])
    
    for name in sorted_keys_t17:
        row = data_t17[name]
        time_str = f"{row['execution_time_mean']:.1f} ± {row['execution_time_std']:.1f}"
        print(f"{name:<10} {row['framework_rank']:<10} {row['observed_rank']:<10} "
              f"{time_str:<15} {row['rank_match']:<10}")

def generate_figures():
    print_header("Generating Figures for Chapter 4")
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    generator = FigureGenerator(output_dir)
    generator.generate_all_figures()
    print(f"\nFigures saved to: {output_dir}")

if __name__ == "__main__":
    generate_tables()
    generate_figures()
    print("\nDone.")
