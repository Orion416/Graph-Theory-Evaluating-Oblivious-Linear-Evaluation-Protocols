# -*- coding: utf-8 -*-
"""Visualization module for generating academic figures."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyBboxPatch, Circle
from matplotlib.lines import Line2D
from scipy import stats
import os
from typing import Dict, Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulators.performance_simulator import PerformanceSimulator
from evaluation.performance_evaluator import PerformanceEvaluator, CorrelationAnalyzer
from evaluation.ranking_analyzer import RankingAnalyzer
from config.experiment_config import ExperimentConfig
from config.protocol_config import ProtocolConfig

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# High quality output settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05


class FigureGenerator:
    """Generator for academic figures."""

    def __init__(self, output_dir: str, random_seed: int = 42):
        self.output_dir = output_dir
        self.simulator = PerformanceSimulator(random_seed)
        self.evaluator = PerformanceEvaluator(random_seed)
        self.ranking_analyzer = RankingAnalyzer(random_seed)
        self.correlation_analyzer = CorrelationAnalyzer(self.evaluator, self.simulator)
        self.config = ExperimentConfig()
        self.protocol_config = ProtocolConfig()

        # Color scheme
        self.protocol_styles = {
            'RLWE': {'color': '#0072B2', 'marker': 'o', 'linestyle': '-'},
            'IKNP': {'color': '#009E73', 'marker': 's', 'linestyle': '--'},
            'Noisy': {'color': '#D55E00', 'marker': '^', 'linestyle': '-.'},
            'PCG': {'color': '#CC79A7', 'marker': 'D', 'linestyle': ':'},
            'HE': {'color': '#F0E442', 'marker': 'p', 'linestyle': '-'},
        }

        # Dimension colors
        self.dimension_colors = {
            'Comm.': '#0072B2',
            'Round': '#D55E00',
            'Comp.': '#009E73',
            'Sec.': '#CC79A7',
        }

        os.makedirs(os.path.join(output_dir, 'svg'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'png'), exist_ok=True)

    def save_figure(self, fig, filename: str):
        """Save figure as SVG and PNG."""
        svg_path = os.path.join(self.output_dir, 'svg', f'{filename}.svg')
        png_path = os.path.join(self.output_dir, 'png', f'{filename}.png')

        fig.savefig(svg_path, format='svg', dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none', pad_inches=0.05)
        fig.savefig(png_path, format='png', dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none', pad_inches=0.05)

        print(f"  Saved: {filename}")
        plt.close(fig)

    def generate_figure7(self):
        """Figure 7: Experimental network topology."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.axis('off')

        # Participant P1
        p1_box = FancyBboxPatch((0.8, 4.5), 2.8, 2.5, boxstyle="round,pad=0.05",
                                 facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=1.5)
        ax.add_patch(p1_box)
        ax.text(2.2, 6.3, 'Participant $P_1$', fontsize=11, fontweight='bold',
                ha='center', va='center', color='#1565C0')
        ax.text(2.2, 5.6, 'Intel Xeon E5-2697 v4', fontsize=9, ha='center')
        ax.text(2.2, 5.2, '18 cores @ 2.3 GHz', fontsize=9, ha='center')
        ax.text(2.2, 4.8, '256 GB DDR4', fontsize=9, ha='center')

        # Participant P2
        p2_box = FancyBboxPatch((8.4, 4.5), 2.8, 2.5, boxstyle="round,pad=0.05",
                                 facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=1.5)
        ax.add_patch(p2_box)
        ax.text(9.8, 6.3, 'Participant $P_2$', fontsize=11, fontweight='bold',
                ha='center', va='center', color='#1565C0')
        ax.text(9.8, 5.6, 'Intel Xeon E5-2697 v4', fontsize=9, ha='center')
        ax.text(9.8, 5.2, '18 cores @ 2.3 GHz', fontsize=9, ha='center')
        ax.text(9.8, 4.8, '256 GB DDR4', fontsize=9, ha='center')

        # Network Emulator
        net_box = FancyBboxPatch((4.6, 4.5), 2.8, 2.5, boxstyle="round,pad=0.05",
                                  facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.5)
        ax.add_patch(net_box)
        ax.text(6, 6.3, 'Network Emulator', fontsize=11, fontweight='bold',
                ha='center', va='center', color='#E65100')
        ax.text(6, 5.6, 'BW: 10 Mbps - 1 Gbps', fontsize=9, ha='center')
        ax.text(6, 5.2, 'Latency: 0.1 - 100 ms', fontsize=9, ha='center')

        # Control Server
        ctrl_box = FancyBboxPatch((4.6, 1.2), 2.8, 2, boxstyle="round,pad=0.05",
                                   facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.5)
        ax.add_patch(ctrl_box)
        ax.text(6, 2.6, 'Control Server', fontsize=11, fontweight='bold',
                ha='center', va='center', color='#2E7D32')
        ax.text(6, 2.0, 'Coordination & Logging', fontsize=9, ha='center')

        # Probes
        probe1 = FancyBboxPatch((1.5, 3.2), 1.4, 0.8, boxstyle="round,pad=0.02",
                                 facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=1)
        ax.add_patch(probe1)
        ax.text(2.2, 3.6, 'Probe', fontsize=8, ha='center', color='#C2185B')

        probe2 = FancyBboxPatch((9.1, 3.2), 1.4, 0.8, boxstyle="round,pad=0.02",
                                 facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=1)
        ax.add_patch(probe2)
        ax.text(9.8, 3.6, 'Probe', fontsize=8, ha='center', color='#C2185B')

        # Connections
        ax.annotate('', xy=(4.6, 5.75), xytext=(3.6, 5.75),
                    arrowprops=dict(arrowstyle='<->', color='#1976D2', lw=2))
        ax.annotate('', xy=(8.4, 5.75), xytext=(7.4, 5.75),
                    arrowprops=dict(arrowstyle='<->', color='#1976D2', lw=2))

        ax.plot([2.2, 2.2], [4.5, 4.0], 'k--', lw=1.2, alpha=0.7)
        ax.plot([9.8, 9.8], [4.5, 4.0], 'k--', lw=1.2, alpha=0.7)
        ax.plot([2.2, 4.6], [3.2, 2.2], color='#2E7D32', ls='--', lw=1.2, alpha=0.6)
        ax.plot([9.8, 7.4], [3.2, 2.2], color='#2E7D32', ls='--', lw=1.2, alpha=0.6)
        ax.plot([6, 6], [4.5, 3.2], color='#2E7D32', ls='--', lw=1.2, alpha=0.6)

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1976D2', label='Participants'),
            mpatches.Patch(facecolor='#FFF3E0', edgecolor='#E65100', label='Network'),
            mpatches.Patch(facecolor='#E8F5E9', edgecolor='#2E7D32', label='Control'),
            mpatches.Patch(facecolor='#FCE4EC', edgecolor='#C2185B', label='Probes'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
                  framealpha=0.95, edgecolor='#CCCCCC')

        plt.tight_layout()
        return fig

    def generate_figure8(self):
        """Figure 8: Parameter space mapping."""
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        protocols_config = self.protocol_config.get_all_protocols()

        for name, params in protocols_config.items():
            style = self.protocol_styles.get(name, {'color': '#333333', 'marker': 'o'})

            if params.ring_dimension > 1:
                field_size = np.log2(params.ring_dimension * params.modulus_bits)
            else:
                field_size = np.log2(params.field_bits)

            security = params.security_bits
            scales = self.config.problem_scales

            x_vals = [field_size] * len(scales)
            y_vals = [security] * len(scales)
            z_vals = scales
            sizes = [60 + (s - 10) * 12 for s in z_vals]

            ax.scatter(x_vals, y_vals, z_vals, c=style['color'],
                      s=sizes, marker=style['marker'], alpha=0.8,
                      edgecolors='white', linewidths=1.2, label=name)
            ax.plot(x_vals, y_vals, z_vals, color=style['color'],
                   linestyle='--', alpha=0.4, linewidth=1.2)

        ax.set_xlabel('Field Size (log$_2$)', fontsize=11, labelpad=8)
        ax.set_ylabel('Security Level (bits)', fontsize=11, labelpad=8)
        ax.set_zlabel('Problem Scale (log$_2$ N)', fontsize=11, labelpad=8)

        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        return fig

    def generate_figure9(self):
        """Figure 9: Communication complexity scaling."""
        scales = self.config.problem_scales
        scale_values = np.array([2**n for n in scales])
        comm_data = self.simulator.generate_table9_data()

        fig, ax = plt.subplots(figsize=(10, 7))

        for protocol, style in self.protocol_styles.items():
            data = comm_data[protocol]
            mean = data['mean']
            std = data['std']

            ax.errorbar(scale_values, mean, yerr=1.96*std,
                       color=style['color'],
                       marker=style['marker'],
                       markersize=7,
                       markerfacecolor=style['color'],
                       markeredgecolor='white',
                       markeredgewidth=1.2,
                       linewidth=2,
                       linestyle=style['linestyle'],
                       capsize=3,
                       capthick=1.2,
                       alpha=0.9,
                       label=protocol,
                       zorder=3)

        x_ref = np.logspace(np.log10(scale_values[0]), np.log10(scale_values[-1]), 100)
        ax.plot(x_ref, x_ref * 0.25, '--', color='#888888', linewidth=1.2, alpha=0.5, zorder=1)
        ax.plot(x_ref, x_ref ** 0.5 * 5, ':', color='#888888', linewidth=1.2, alpha=0.5, zorder=1)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Problem Scale $N$ (OLE instances)', fontsize=12)
        ax.set_ylabel('Communication (KB)', fontsize=12)
        ax.set_xticks(scale_values)
        ax.set_xticklabels([f'$2^{{{s}}}$' for s in scales], fontsize=10)
        ax.set_ylim(1, 5e5)

        ax.grid(True, which='major', linestyle='-', alpha=0.25, color='#CCCCCC')
        ax.grid(True, which='minor', linestyle=':', alpha=0.15, color='#DDDDDD')

        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')
            spine.set_linewidth(0.8)

        ax.legend(loc='upper left', fontsize=9, framealpha=0.95,
                  edgecolor='#CCCCCC', ncol=2)
        ax.set_facecolor('white')

        plt.tight_layout()
        return fig

    def generate_figure10(self):
        """Figure 10: Computation-Communication Trade-off."""
        scales = self.config.problem_scales
        comm_data = self.simulator.generate_table9_data()
        comp_data = self.simulator.generate_table11_data()

        fig, ax = plt.subplots(figsize=(10, 8))

        all_points = []

        for protocol, style in self.protocol_styles.items():
            comm_vals = comm_data[protocol]['mean']
            comp_vals = comp_data[protocol]['mean']
            base_color = np.array(plt.cm.colors.to_rgb(style['color']))

            for i, (comm, comp) in enumerate(zip(comm_vals, comp_vals)):
                intensity = 0.4 + 0.6 * (i / (len(scales) - 1))
                color = base_color * intensity + np.array([1, 1, 1]) * (1 - intensity) * 0.2
                size = 70 + i * 25

                ax.scatter(comm, comp, c=[color], s=size, marker=style['marker'],
                          edgecolors='white', linewidths=1.2, alpha=0.85, zorder=3)
                all_points.append([np.log10(comm), np.log10(comp)])

        # Pareto front
        all_points = np.array(all_points)
        pareto_mask = np.ones(len(all_points), dtype=bool)
        for i in range(len(all_points)):
            for j in range(len(all_points)):
                if i != j:
                    if (all_points[j, 0] <= all_points[i, 0] and
                        all_points[j, 1] <= all_points[i, 1] and
                        (all_points[j, 0] < all_points[i, 0] or all_points[j, 1] < all_points[i, 1])):
                        pareto_mask[i] = False
                        break

        pareto_points = all_points[pareto_mask]
        sorted_indices = np.argsort(pareto_points[:, 0])
        pareto_points = pareto_points[sorted_indices]

        pareto_comm = 10 ** pareto_points[:, 0]
        pareto_comp = 10 ** pareto_points[:, 1]
        ax.plot(pareto_comm, pareto_comp, '--', color='#E74C3C', linewidth=2, alpha=0.7, zorder=2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Communication (KB)', fontsize=12)
        ax.set_ylabel('Computation ($\\times 10^6$ ops)', fontsize=12)
        ax.set_xlim(1, 5e5)
        ax.set_ylim(0.1, 1e3)

        ax.grid(True, which='major', linestyle='-', alpha=0.25, color='#CCCCCC')
        ax.grid(True, which='minor', linestyle=':', alpha=0.15, color='#DDDDDD')

        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')
            spine.set_linewidth(0.8)

        legend_handles = [
            Line2D([0], [0], marker=style['marker'], color='w',
                   markerfacecolor=style['color'], markersize=8,
                   markeredgecolor='white', markeredgewidth=1, label=protocol)
            for protocol, style in self.protocol_styles.items()
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
                  framealpha=0.95, edgecolor='#CCCCCC')
        ax.set_facecolor('white')

        plt.tight_layout()
        return fig

    def generate_figure11(self):
        """Figure 11: Radar chart comparison."""
        radar_data = self.evaluator.generate_radar_data(scale=16)

        categories = ['Comm.', 'Round', 'Comp.', 'Sec.']
        n_categories = len(categories)

        angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        linestyles = ['-', '--', '-.', ':', '-']

        for idx, (protocol, style) in enumerate(self.protocol_styles.items()):
            values = radar_data[protocol]
            values = np.concatenate([values, [values[0]]])

            ax.plot(angles, values,
                    color=style['color'],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2,
                    alpha=0.85,
                    marker=style['marker'],
                    markersize=6,
                    markerfacecolor=style['color'],
                    markeredgecolor='white',
                    markeredgewidth=1,
                    label=protocol,
                    zorder=3)
            ax.fill(angles, values, color=style['color'], alpha=0.06, zorder=1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='#666666')
        ax.grid(True, linestyle='-', alpha=0.3, color='#CCCCCC', linewidth=0.8)
        ax.set_rlabel_position(30)

        ax.legend(loc='upper right', fontsize=9, framealpha=0.95,
                  edgecolor='#CCCCCC', bbox_to_anchor=(1.2, 1.0))
        ax.set_facecolor('white')

        plt.tight_layout()
        return fig

    def generate_figure12(self):
        """Figure 12: Weight sensitivity analysis."""
        sensitivity_data = self.evaluator.generate_sensitivity_data(scale=16)

        weight_configs = ['Base', 'Comm.', 'Comp.', 'Bal.', 'Sec.']
        n_configs = len(weight_configs)

        fig, ax = plt.subplots(figsize=(11, 7))
        x_positions = np.arange(n_configs)

        for x in x_positions:
            ax.axvline(x=x, color='#EEEEEE', linewidth=1.5, zorder=1)

        for protocol, style in self.protocol_styles.items():
            scores = sensitivity_data[protocol]

            ax.plot(x_positions, scores,
                   color=style['color'],
                   linewidth=2,
                   alpha=0.85,
                   zorder=2)
            ax.scatter(x_positions, scores,
                      c=style['color'],
                      s=80,
                      marker=style['marker'],
                      edgecolors='white',
                      linewidths=1.5,
                      zorder=3,
                      label=protocol)

        ax.set_xlim(-0.3, n_configs - 0.7)
        ax.set_ylim(0.1, 0.95)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(weight_configs, fontsize=11)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.yaxis.grid(True, linestyle='-', alpha=0.25, color='#CCCCCC')

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#CCCCCC')
            spine_width = 0.8

        ax.legend(loc='center right', fontsize=9, framealpha=0.95,
                  edgecolor='#CCCCCC', bbox_to_anchor=(1.12, 0.5))
        ax.set_facecolor('white')

        plt.tight_layout()
        return fig

    def generate_figure13(self):
        """Figure 13: Score decomposition."""
        decomp_data = self.evaluator.generate_decomposition_data(scale=16)

        protocols = list(decomp_data.keys())
        n_protocols = len(protocols)

        fig, ax = plt.subplots(figsize=(10, 7))
        x_positions = np.arange(n_protocols)
        bar_width = 0.6

        bottoms = np.zeros(n_protocols)
        dimension_names = ['Comm.', 'Round', 'Comp.', 'Sec.']

        for dim_idx, dim_name in enumerate(dimension_names):
            values = np.array([decomp_data[p][dim_idx] for p in protocols])
            color = self.dimension_colors[dim_name]

            ax.bar(x_positions, values, bar_width,
                  bottom=bottoms,
                  color=color,
                  edgecolor='white',
                  linewidth=1.2,
                  alpha=0.9,
                  label=dim_name,
                  zorder=3)
            bottoms += values

        for i, protocol in enumerate(protocols):
            total_score = np.sum(decomp_data[protocol])
            ax.text(i, total_score + 0.02, f'{total_score:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   color='#333333')

        ax.set_xlim(-0.5, n_protocols - 0.5)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(protocols, fontsize=11)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.grid(True, linestyle='-', alpha=0.25, color='#CCCCCC', zorder=0)
        ax.set_axisbelow(True)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#CCCCCC')

        ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='#CCCCCC')
        ax.set_facecolor('white')

        plt.tight_layout()
        return fig

    def generate_figure14(self):
        """Figure 14: OLE vs OT Comparison."""
        comparison_data = self.simulator.generate_table15_data(scale=16)

        protocols = list(comparison_data.keys())
        n_protocols = len(protocols)

        comm_values = np.array([comparison_data[p]['communication'] for p in protocols])
        comp_values = np.array([comparison_data[p]['computation'] for p in protocols])

        colors = ['#0072B2' if comparison_data[p]['type'] == 'OLE' else '#D55E00' for p in protocols]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        x_positions = np.arange(n_protocols)
        bar_width = 0.65

        # Communication
        ax1.bar(x_positions, comm_values, bar_width,
                color=colors, edgecolor='white', linewidth=1.2, alpha=0.9, zorder=3)
        ax1.set_yscale('log')
        ax1.set_ylabel('Communication (KB)', fontsize=11)
        ax1.set_xticks(x_positions)
        protocol_labels = [p.replace('-OLE', '').replace(' OT', '') for p in protocols]
        ax1.set_xticklabels(protocol_labels, fontsize=9, rotation=45, ha='right')
        ax1.set_ylim(1, 1e5)
        ax1.yaxis.grid(True, linestyle='-', alpha=0.25, color='#CCCCCC', zorder=0)
        ax1.set_axisbelow(True)
        ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')

        for spine in ['top', 'right']:
            ax1.spines[spine].set_visible(False)
        ax1.set_facecolor('white')

        # Computation
        ax2.bar(x_positions, comp_values, bar_width,
                color=colors, edgecolor='white', linewidth=1.2, alpha=0.9, zorder=3)
        ax2.set_yscale('log')
        ax2.set_ylabel('Computation ($\\times 10^6$ ops)', fontsize=11)
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(protocol_labels, fontsize=9, rotation=45, ha='right')
        ax2.set_ylim(0.1, 1e3)
        ax2.yaxis.grid(True, linestyle='-', alpha=0.25, color='#CCCCCC', zorder=0)
        ax2.set_axisbelow(True)
        ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')

        for spine in ['top', 'right']:
            ax2.spines[spine].set_visible(False)
        ax2.set_facecolor('white')

        # Legend
        legend_handles = [
            mpatches.Patch(facecolor='#0072B2', edgecolor='white', label='OLE'),
            mpatches.Patch(facecolor='#D55E00', edgecolor='white', label='OT'),
        ]
        ax2.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.95)

        plt.tight_layout()
        return fig

    def generate_figure15(self):
        """Figure 15: Ranking Inconsistency (Alluvial Diagram)."""
        rankings_data = self.ranking_analyzer.generate_figure15_data(scale=16)

        methods = ['Naive\nComm.', 'Naive\nComp.', 'Naive\nComb.', 'Framework']
        n_methods = len(methods)

        protocol_colors = {p: s['color'] for p, s in self.protocol_styles.items()}

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-0.3, n_methods + 0.3)
        ax.set_ylim(0, 6)

        x_positions = np.arange(n_methods)

        for x in x_positions:
            ax.axvline(x=x, color='#DDDDDD', linewidth=1.5, zorder=1)

        flow_width = 0.3

        for protocol, ranks in rankings_data.items():
            color = protocol_colors[protocol]
            y_coords = 6 - ranks

            for i in range(n_methods - 1):
                x1, x2 = x_positions[i], x_positions[i + 1]
                y1, y2 = y_coords[i], y_coords[i + 1]

                n_points = 50
                x_curve = np.linspace(x1, x2, n_points)
                y_top = np.linspace(y1 + flow_width/2, y2 + flow_width/2, n_points)
                y_bottom = np.linspace(y1 - flow_width/2, y2 - flow_width/2, n_points)

                polygon_x = np.concatenate([x_curve, x_curve[::-1]])
                polygon_y = np.concatenate([y_top, y_bottom[::-1]])

                polygon = Polygon(list(zip(polygon_x, polygon_y)),
                                facecolor=color, edgecolor='white',
                                linewidth=0.3, alpha=0.65, zorder=2)
                ax.add_patch(polygon)

            # Nodes
            for i, (x, y) in enumerate(zip(x_positions, y_coords)):
                node = Circle((x, y), 0.18, facecolor=color, edgecolor='white',
                             linewidth=1.5, zorder=4)
                ax.add_patch(node)
                ax.text(x, y, str(int(ranks[i])), fontsize=10, fontweight='bold',
                       ha='center', va='center', color='white', zorder=5)

        for i, method in enumerate(methods):
            ax.text(x_positions[i], -0.4, method, fontsize=10,
                   ha='center', va='top', color='#333333')

        for rank in range(1, 6):
            y = 6 - rank
            ax.text(-0.25, y, str(rank), fontsize=9, ha='right', va='center', color='#666666')

        legend_handles = [
            mpatches.Patch(facecolor=color, edgecolor='white', linewidth=1, label=protocol)
            for protocol, color in protocol_colors.items()
        ]
        ax.legend(handles=legend_handles, loc='center right', fontsize=9,
                  framealpha=0.95, edgecolor='#CCCCCC', bbox_to_anchor=(1.15, 0.5))

        ax.axis('off')
        fig.patch.set_facecolor('white')

        plt.tight_layout()
        return fig

    def generate_figure16(self):
        """Figure 16: Framework validation correlation analysis."""
        corr_data = self.correlation_analyzer.compute_prediction_correlation(scale=16)

        protocols = corr_data['protocols']
        scores = corr_data['scores']
        times = corr_data['times']
        time_stds = corr_data['time_stds']

        performance = 1000 / times
        performance_std = performance * (time_stds / times)

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, protocol in enumerate(protocols):
            style = self.protocol_styles[protocol]
            ax.errorbar(scores[i], performance[i],
                       yerr=performance_std[i],
                       color=style['color'],
                       marker=style['marker'],
                       markersize=12,
                       markerfacecolor=style['color'],
                       markeredgecolor='white',
                       markeredgewidth=1.5,
                       capsize=5,
                       capthick=1.5,
                       elinewidth=1.5,
                       alpha=0.9,
                       zorder=3,
                       label=protocol)

        # Regression line
        log_performance = np.log10(performance)
        slope, intercept, r_value, p_value, std_err = stats.linregress(scores, log_performance)

        x_min = max(0, min(scores) - 0.1)
        x_max = min(1.0, max(scores) + 0.1)

        x_line = np.linspace(x_min, x_max, 100)
        y_line = 10 ** (slope * x_line + intercept)

        ax.plot(x_line, y_line, '--', color='#E74C3C', linewidth=2, alpha=0.7, zorder=2)

        # Confidence interval
        n = len(scores)
        x_mean = np.mean(scores)
        denominator = np.sum((scores - x_mean)**2)
        if denominator > 0:
            se = std_err * np.sqrt(1/n + (x_line - x_mean)**2 / denominator)
            t_val = stats.t.ppf(0.975, max(1, n-2))
            y_upper = 10 ** (slope * x_line + intercept + t_val * se)
            y_lower = 10 ** (slope * x_line + intercept - t_val * se)
            ax.fill_between(x_line, y_lower, y_upper, color='#E74C3C', alpha=0.12, zorder=1)

        ax.text(0.05, 0.95, f'$r$ = {r_value:.3f}',
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               color='#E74C3C', verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#E74C3C'))

        ax.set_xlabel('Predicted Score', fontsize=12)
        ax.set_ylabel('Performance (1000 / Time)', fontsize=12)
        ax.set_xlim(x_min, x_max)
        ax.set_yscale('log')

        ax.grid(True, which='major', linestyle='-', alpha=0.25, color='#CCCCCC', zorder=0)
        ax.grid(True, which='minor', linestyle=':', alpha=0.15, color='#DDDDDD', zorder=0)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#CCCCCC')

        legend_handles = [
            Line2D([0], [0], marker=style['marker'], color='w',
                   markerfacecolor=style['color'], markersize=9,
                   markeredgecolor='white', markeredgewidth=1, label=protocol)
            for protocol, style in self.protocol_styles.items()
        ]
        ax.legend(handles=legend_handles, loc='lower right', fontsize=9,
                  framealpha=0.95, edgecolor='#CCCCCC')
        ax.set_facecolor('white')

        plt.tight_layout()
        return fig

    def generate_all_figures(self):
        """Generate all figures."""
        print("Generating figures (600 DPI, publication quality)...")

        print("\n  Figure 8: Network Topology...")
        fig8 = self.generate_figure7()
        self.save_figure(fig8, 'Figure 8-Experimental Network Topology')

        print("  Figure 9: Parameter Space Mapping...")
        fig9 = self.generate_figure8()
        self.save_figure(fig9, 'Figure 9-Parameter Space Mapping')

        print("  Figure 10: Communication Complexity Scaling...")
        fig10 = self.generate_figure9()
        self.save_figure(fig10, 'Figure 10-Communication Complexity Scaling')

        print("  Figure 11: Computation-Communication Trade-off...")
        fig11 = self.generate_figure10()
        self.save_figure(fig11, 'Figure 11-Computation-Communication Trade-off')

        print("  Figure 12: Radar Chart Comparison...")
        fig12 = self.generate_figure11()
        self.save_figure(fig12, 'Figure 12-Radar Chart Comparison')

        print("  Figure 13: Weight Sensitivity Analysis...")
        fig13 = self.generate_figure12()
        self.save_figure(fig13, 'Figure 13-Weight Sensitivity Analysis')

        print("  Figure 14: Score Decomposition...")
        fig14 = self.generate_figure13()
        self.save_figure(fig14, 'Figure 14-Score Decomposition')

        print("  Figure 15: OLE vs OT Comparison...")
        fig15 = self.generate_figure14()
        self.save_figure(fig15, 'Figure 15-OLE vs OT Comparison')

        print("  Figure 16: Ranking Inconsistency...")
        fig16 = self.generate_figure15()
        self.save_figure(fig16, 'Figure 16-Ranking Inconsistency')

        print("  Figure 17: Framework Validation...")
        fig17 = self.generate_figure16()
        self.save_figure(fig17, 'Figure 17-Framework Validation')

        print("\nAll figures generated successfully!")
