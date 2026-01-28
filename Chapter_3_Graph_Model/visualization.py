# -*- coding: utf-8 -*-
"""Visualization Module for OLE Protocol Analysis"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch, Rectangle
from matplotlib.collections import LineCollection
import os
from typing import Dict, List, Optional, Tuple

from protocols import Protocol
from graph_model import CommunicationGraph
from metrics import GraphMetrics
from validation import SelectionResult, ExternalValidationSummary

# Set matplotlib style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# Academic color scheme
COLORS = {
    # Theme colors
    'primary': '#2E4057',        # Dark blue-gray
    'secondary': '#048A81',      # Teal
    'accent': '#C84630',         # Dark red
    'highlight': '#8B5E3C',      # Brown

    # Protocol arrows
    'arrow_ab': '#2E86AB',       # Blue (A->B)
    'arrow_ba': '#A23B72',       # Magenta (B->A)

    # Party nodes
    'party_a': '#1B4965',        # Dark blue
    'party_b': '#5FA8D3',        # Light blue

    # Crypto foundations
    'he': '#2E86AB',             # HE - Blue
    'ot': '#E07A5F',             # OT Extension - Orange-red
    'algebraic': '#81B29A',      # Algebraic - Green
    'sublinear': '#9B59B6',      # Sublinear - Purple
    'baseline': '#6C757D',       # Baseline - Gray

    # Auxiliary colors
    'grid': '#E8E8E8',           # Grid lines
    'background': '#FFFFFF',     # Background
    'text': '#2C3E50',           # Text
}


def create_figure1(protocols: Dict[str, Protocol],
                   graphs: Dict[str, CommunicationGraph]) -> plt.Figure:
    """Figure 1: Temporal Communication Patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # Select 4 protocols
    selected = ['P1', 'P4', 'P5', 'P7']

    for idx, pid in enumerate(selected):
        ax = axes[idx]
        ax.set_facecolor(COLORS['background'])

        protocol = protocols[pid]
        graph = graphs[pid]

        # Draw party lines
        ax.axhline(y=0.75, color=COLORS['party_a'], linewidth=2.5, alpha=0.9)
        ax.axhline(y=0.25, color=COLORS['party_b'], linewidth=2.5, alpha=0.9)

        # Party labels
        ax.text(-0.08, 0.75, r'$v_1$', fontsize=12, fontweight='bold',
                va='center', ha='right', color=COLORS['party_a'])
        ax.text(-0.08, 0.25, r'$v_2$', fontsize=12, fontweight='bold',
                va='center', ha='right', color=COLORS['party_b'])

        num_rounds = graph.num_rounds

        # Draw round separators
        for i in range(1, num_rounds + 1):
            ax.axvline(x=i, color=COLORS['grid'], linestyle='--',
                       linewidth=1, alpha=0.6)

        # Max weight for normalization
        max_weight = max(msg.size_bits for msg in protocol.messages)

        # Draw arrows for messages
        for msg in protocol.messages:
            # Calculate time position
            round_msgs = [m for m in protocol.messages if m.round_num == msg.round_num]
            msg_idx = round_msgs.index(msg)
            time = msg.round_num - 0.5 + (msg_idx + 0.5) / (len(round_msgs) + 1)

            # Determine direction and color
            if msg.sender == 0:
                start_y, end_y = 0.75, 0.25
                color = COLORS['arrow_ab']
            else:
                start_y, end_y = 0.25, 0.75
                color = COLORS['arrow_ba']

            # Arrow width proportional to size
            norm_weight = msg.size_bits / max_weight
            arrow_width = 1.0 + norm_weight * 3.5

            ax.annotate('',
                        xy=(time, end_y),
                        xytext=(time, start_y),
                        arrowprops=dict(
                            arrowstyle='->,head_width=0.35,head_length=0.25',
                            color=color,
                            linewidth=arrow_width,
                            alpha=0.85,
                            shrinkA=4,
                            shrinkB=4,
                        ))

        # Set axes
        ax.set_xlim(-0.15, num_rounds + 0.15)
        ax.set_ylim(0, 1)

        # X-axis labels
        round_ticks = np.arange(0.5, num_rounds + 0.5, 1)
        round_labels = [f'R{i+1}' for i in range(num_rounds)]
        ax.set_xticks(round_ticks)
        ax.set_xticklabels(round_labels, fontsize=10)

        ax.set_yticks([])

        # Subplot label
        ax.text(0.5, -0.08, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', ha='center', va='top')

        # Simplify spines
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color(COLORS['grid'])
        ax.spines['bottom'].set_linewidth(0.8)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.12)

    return fig


def create_figure2(graphs: Dict[str, CommunicationGraph]) -> plt.Figure:
    """Figure 2: Communication Graph Structures"""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    selected = ['P1', 'P4', 'P6', 'P8']

    for idx, pid in enumerate(selected):
        ax = axes[idx]
        ax.set_facecolor(COLORS['background'])

        graph = graphs[pid]

        # Node positions
        v1_pos = (0.5, 0.82)
        v2_pos = (0.5, 0.18)

        # Separate edges
        edges_down = [e for e in graph.edges if e[0] == 0]
        edges_up = [e for e in graph.edges if e[0] == 1]

        # Normalize weights
        all_weights = [e[2] for e in graph.edges]
        log_weights = np.log2(np.array(all_weights) + 1)
        max_log = max(log_weights) if len(log_weights) > 0 else 1
        min_log = min(log_weights) if len(log_weights) > 0 else 0

        def normalize_weight(w):
            log_w = np.log2(w + 1)
            if max_log == min_log:
                return 2.5
            normalized = (log_w - min_log) / (max_log - min_log)
            return 1.0 + normalized * 4.0

        # Draw v1->v2 edges
        n_down = len(edges_down)
        if n_down > 0:
            offsets = np.linspace(-0.12, 0.12, n_down) if n_down > 1 else [0]
            for i, (_, _, weight) in enumerate(edges_down):
                lw = normalize_weight(weight)
                start_x = v1_pos[0] + offsets[i] - 0.06
                end_x = v2_pos[0] + offsets[i] - 0.06

                style = f"Simple,tail_width={lw/18},head_width={lw/6},head_length={lw/10}"
                arrow = FancyArrowPatch(
                    (start_x, v1_pos[1] - 0.06),
                    (end_x, v2_pos[1] + 0.06),
                    connectionstyle="arc3,rad=-0.18",
                    arrowstyle=style,
                    color=COLORS['arrow_ab'],
                    alpha=0.8,
                    linewidth=lw,
                    zorder=1
                )
                ax.add_patch(arrow)

        # Draw v2->v1 edges
        n_up = len(edges_up)
        if n_up > 0:
            offsets = np.linspace(-0.12, 0.12, n_up) if n_up > 1 else [0]
            for i, (_, _, weight) in enumerate(edges_up):
                lw = normalize_weight(weight)
                start_x = v2_pos[0] + offsets[i] + 0.06
                end_x = v1_pos[0] + offsets[i] + 0.06

                style = f"Simple,tail_width={lw/18},head_width={lw/6},head_length={lw/10}"
                arrow = FancyArrowPatch(
                    (start_x, v2_pos[1] + 0.06),
                    (end_x, v1_pos[1] - 0.06),
                    connectionstyle="arc3,rad=-0.18",
                    arrowstyle=style,
                    color=COLORS['arrow_ba'],
                    alpha=0.8,
                    linewidth=lw,
                    zorder=1
                )
                ax.add_patch(arrow)

        # Draw nodes
        node_radius = 0.07

        v1_circle = Circle(v1_pos, node_radius, facecolor=COLORS['party_a'],
                           edgecolor=COLORS['text'], linewidth=2, zorder=3)
        ax.add_patch(v1_circle)
        ax.text(v1_pos[0], v1_pos[1], r'$v_1$', fontsize=13, fontweight='bold',
                ha='center', va='center', color='white', zorder=4)

        v2_circle = Circle(v2_pos, node_radius, facecolor=COLORS['party_b'],
                           edgecolor=COLORS['text'], linewidth=2, zorder=3)
        ax.add_patch(v2_circle)
        ax.text(v2_pos[0], v2_pos[1], r'$v_2$', fontsize=13, fontweight='bold',
                ha='center', va='center', color='white', zorder=4)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Subplot label
        ax.text(0.5, 0.02, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', ha='center', va='top')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.08)

    return fig


def create_figure3(metrics: Dict[str, GraphMetrics]) -> plt.Figure:
    """Figure 3: Radar Chart Comparison"""
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    # Metric dimensions
    categories = ['|E|', 'W', r'$\beta$', r'$G_w$', 'H']
    n_categories = len(categories)

    # Compute normalization params
    all_values = {
        'edge_count': [m.num_edges for m in metrics.values()],
        'total_weight': [m.total_weight for m in metrics.values()],
        'balance': [m.edge_direction_balance for m in metrics.values()],
        'concentration': [m.weight_concentration_index for m in metrics.values()],
        'entropy': [m.temporal_entropy for m in metrics.values()],
    }

    min_vals = {k: min(v) for k, v in all_values.items()}
    max_vals = {k: max(v) for k, v in all_values.items()}

    def normalize(m):
        values = []
        for key in ['edge_count', 'total_weight', 'balance', 'concentration', 'entropy']:
            if key == 'edge_count':
                val = m.num_edges
            elif key == 'total_weight':
                val = m.total_weight
            elif key == 'balance':
                val = m.edge_direction_balance
            elif key == 'concentration':
                val = m.weight_concentration_index
            else:
                val = m.temporal_entropy

            range_val = max_vals[key] - min_vals[key]
            if range_val > 0:
                values.append((val - min_vals[key]) / range_val)
            else:
                values.append(0.5)
        return values

    # Angles
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]

    # Protocol styles
    styles = {
        'P1': {'color': COLORS['he'], 'marker': 'o', 'linestyle': '-'},
        'P4': {'color': COLORS['ot'], 'marker': 's', 'linestyle': '--'},
        'P8': {'color': COLORS['sublinear'], 'marker': '^', 'linestyle': ':'},
        'P0': {'color': COLORS['baseline'], 'marker': 'd', 'linestyle': '-.'},
    }
    selected_protocols = ['P1', 'P4', 'P8', 'P0']

    for pid in selected_protocols:
        m = metrics[pid]
        values = normalize(m)
        values = values + [values[0]]

        style = styles[pid]
        ax.plot(angles, values,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=2.2,
                alpha=0.9,
                marker=style['marker'],
                markersize=7,
                markerfacecolor=style['color'],
                markeredgecolor='white',
                markeredgewidth=1.2,
                zorder=3)

        ax.fill(angles, values, color=style['color'], alpha=0.06, zorder=1)

    # Set axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')

    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''], fontsize=9)

    ax.grid(True, linestyle='-', alpha=0.25, color=COLORS['grid'], linewidth=0.8)
    ax.set_rlabel_position(30)
    ax.tick_params(axis='x', pad=12)
    ax.set_facecolor('#FAFAFA')

    # Symbol legend
    legend_x = 1.25
    legend_y_start = 0.9
    legend_spacing = 0.1

    for i, pid in enumerate(selected_protocols):
        style = styles[pid]
        y_pos = legend_y_start - i * legend_spacing
        ax.scatter([legend_x], [y_pos], marker=style['marker'], s=80,
                   c=style['color'], edgecolor='white', linewidth=1.2,
                   transform=ax.transAxes, zorder=10, clip_on=False)
        ax.plot([legend_x - 0.05, legend_x + 0.05], [y_pos, y_pos],
                color=style['color'], linestyle=style['linestyle'],
                linewidth=2, transform=ax.transAxes, zorder=9, clip_on=False)
        # Show ID only
        ax.text(legend_x + 0.1, y_pos, pid, fontsize=10,
                va='center', ha='left', color=COLORS['text'],
                transform=ax.transAxes)

    plt.tight_layout()

    return fig


def create_figure4(metrics: Dict[str, GraphMetrics],
                   protocols: Dict[str, Protocol]) -> plt.Figure:
    """Figure 4: Protocol Distribution Scatter Plot"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Category styles
    cat_styles = {
        'HE': {'color': COLORS['he'], 'marker': 'o'},
        'OT': {'color': COLORS['ot'], 'marker': 's'},
        'Algebraic': {'color': COLORS['algebraic'], 'marker': '^'},
        'Sublinear': {'color': COLORS['sublinear'], 'marker': 'D'},
        'Baseline': {'color': COLORS['baseline'], 'marker': '*'},
    }

    # Collect data
    for pid, m in metrics.items():
        cat = protocols[pid].foundation.value
        style = cat_styles[cat]

        # Size based on entropy
        size = 120 + m.temporal_entropy * 150

        ax.scatter(m.total_weight, m.num_edges,
                   c=style['color'], s=size, marker=style['marker'],
                   edgecolors='white', linewidths=1.5, alpha=0.85, zorder=3)

        # Protocol ID labels
        offset_x = 150
        offset_y = 0.2
        if pid == 'P2':
            offset_x = 200
            offset_y = -0.3
        elif pid == 'P0':
            offset_x = -300
            offset_y = 0.3

        ax.annotate(pid, (m.total_weight, m.num_edges),
                    xytext=(m.total_weight + offset_x, m.num_edges + offset_y),
                    fontsize=10, fontweight='bold', color=COLORS['text'],
                    ha='left' if offset_x > 0 else 'right', zorder=4)

    # Iso-product curves
    x_line = np.linspace(800, 13000, 100)
    for product in [25000, 45000]:
        y_line = product / x_line
        mask = (y_line >= 3) & (y_line <= 11)
        ax.plot(x_line[mask], y_line[mask], '--',
                color='#AAAAAA', linewidth=1.2, alpha=0.5, zorder=1)

    # Axes
    ax.set_xlabel('W (bits)', fontsize=12, fontweight='bold')
    ax.set_ylabel('|E|', fontsize=12, fontweight='bold')

    ax.set_xlim(0, 13500)
    ax.set_ylim(3, 11)

    ax.grid(True, linestyle='-', alpha=0.15, color=COLORS['grid'], zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(0.8)

    # Symbol legend
    legend_elements = []
    for cat, style in cat_styles.items():
        legend_elements.append(
            plt.scatter([], [], c=style['color'], s=100, marker=style['marker'],
                        edgecolors='white', linewidths=1.5, label=cat)
        )

    ax.legend(handles=legend_elements, loc='lower right',
              framealpha=0.95, edgecolor=COLORS['grid'], fontsize=9)

    ax.set_facecolor('#FAFAFA')
    plt.tight_layout()

    return fig


def compute_preservation_matrix() -> np.ndarray:
    """Compute preservation matrix."""
    # Define formal capabilities
    # [NL, PC, AN, GM]
    can_express_direction = np.array([0, 1, 0, 1])    # Express direction
    can_express_size = np.array([0, 0, 0, 1])         # Express precise size
    can_express_temporal = np.array([0, 1, 0, 1])     # Express temporal
    can_express_rounds = np.array([0, 1, 0, 1])       # Express rounds
    has_quantitative = np.array([0, 0, 0, 1])         # Quantitative
    has_structure = np.array([0, 1, 0, 1])            # Structure

    # Preservation scores
    # Direction = direction * 0.7 + structure * 0.3
    directions_score = can_express_direction * 0.7 + has_structure * 0.3

    # Size = size * 0.8 + quantitative * 0.2
    sizes_score = can_express_size * 0.8 + has_quantitative * 0.2

    # Temporal = temporal
    temporal_score = can_express_temporal

    # Rounds = rounds * 0.6 + structure * 0.4
    round_score = can_express_rounds * 0.6 + has_structure * 0.4

    preservation_matrix = np.array([
        directions_score,
        sizes_score,
        temporal_score,
        round_score
    ], dtype=float)

    # Normalize (Graph Model = 1.0)
    for i in range(4):
        if preservation_matrix[i, 3] > 0:
            preservation_matrix[i] = preservation_matrix[i] / preservation_matrix[i, 3]

    return preservation_matrix


def create_figure5() -> plt.Figure:
    """Figure 5: Information Preservation Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Method and Aspect labels
    methods = ['NL', 'PC', 'AN', 'GM']
    aspects = ['Dir.', 'Size', 'Temp.', 'Round']

    # Compute matrix
    preservation_matrix = compute_preservation_matrix()

    n_methods = len(methods)
    n_aspects = len(aspects)

    cell_width = 0.9
    cell_height = 0.7
    gap_x = 0.15
    gap_y = 0.15

    start_x = 1.2
    start_y = 0.4

    # Color map
    def get_color(value):
        # Blue gradient
        if value < 0.25:
            return '#E8F4F8'
        elif value < 0.45:
            return '#B8D4E3'
        elif value < 0.65:
            return '#7EB5CF'
        elif value < 0.85:
            return '#4A96B8'
        else:
            return '#1B4965'

    # Draw cells
    for i in range(n_aspects):
        for j in range(n_methods):
            value = preservation_matrix[i, j]
            x = start_x + j * (cell_width + gap_x)
            y = start_y + (n_aspects - 1 - i) * (cell_height + gap_y)
            color = get_color(value)

            rect = FancyBboxPatch(
                (x, y), cell_width, cell_height,
                boxstyle='round,pad=0.02,rounding_size=0.06',
                facecolor=color, edgecolor='#CCCCCC', linewidth=1, zorder=2
            )
            ax.add_patch(rect)

    # Method labels
    for j, method in enumerate(methods):
        x = start_x + j * (cell_width + gap_x) + cell_width / 2
        y = start_y + n_aspects * (cell_height + gap_y) + 0.08
        ax.text(x, y, method, ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=COLORS['text'])

    # Aspect labels
    for i, aspect in enumerate(aspects):
        x = start_x - 0.12
        y = start_y + (n_aspects - 1 - i) * (cell_height + gap_y) + cell_height / 2
        ax.text(x, y, aspect, ha='right', va='center',
                fontsize=11, fontweight='bold', color=COLORS['text'])

    # Color bar
    total_width = n_methods * (cell_width + gap_x) - gap_x
    total_height = n_aspects * (cell_height + gap_y) - gap_y

    cbar_x = start_x + total_width + 0.5
    cbar_y = start_y
    cbar_width = 0.2
    cbar_height = total_height

    n_grad = 50
    for k in range(n_grad):
        frac = k / n_grad
        y_pos = cbar_y + frac * cbar_height
        color = get_color(frac)
        rect = Rectangle((cbar_x, y_pos), cbar_width, cbar_height / n_grad * 1.1,
                          facecolor=color, edgecolor='none')
        ax.add_patch(rect)

    border = Rectangle((cbar_x, cbar_y), cbar_width, cbar_height,
                        facecolor='none', edgecolor='#CCCCCC', linewidth=1)
    ax.add_patch(border)

    # Color bar labels
    ax.text(cbar_x + cbar_width + 0.08, cbar_y, '0', ha='left', va='center',
            fontsize=9, color='#666666')
    ax.text(cbar_x + cbar_width + 0.08, cbar_y + cbar_height, '1', ha='left', va='center',
            fontsize=9, color='#666666')

    ax.set_xlim(0, start_x + total_width + 1.2)
    ax.set_ylim(0, start_y + n_aspects * (cell_height + gap_y) + 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    return fig


def create_figure6(selection_results: List[SelectionResult]) -> plt.Figure:
    """Figure 6: Protocol Selection Accuracy"""
    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = np.array([r.scenario_id for r in selection_results])
    scores = np.array([r.agreement_score for r in selection_results])

    # Color by score
    colors = []
    for score in scores:
        if score >= 0.9:
            colors.append(COLORS['primary'])      # Full agreement
        elif score >= 0.4:
            colors.append(COLORS['secondary'])    # Partial agreement
        else:
            colors.append(COLORS['accent'])       # Disagreement

    bar_width = 0.6
    bars = ax.bar(scenarios, scores, width=bar_width,
                  color=colors, edgecolor='white', linewidth=1.2,
                  alpha=0.9, zorder=3)

    # Threshold line
    ax.axhline(y=0.8, color=COLORS['accent'], linestyle='--',
               linewidth=2, alpha=0.7, zorder=2)

    # Axes
    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Agreement', fontsize=12, fontweight='bold', labelpad=8)

    ax.set_xticks(scenarios)
    ax.set_xticklabels([f'S{i}' for i in scenarios], fontsize=10)

    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(0.2, 10.8)

    # Grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.2, color=COLORS['grid'], zorder=0)
    ax.set_axisbelow(True)

    # Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(COLORS['grid'])
        ax.spines[spine].set_linewidth(0.8)

    # Statistical summary
    avg_score = np.mean(scores)
    full_agree = sum(1 for s in scores if s >= 0.9)

    ax.text(0.03, 0.97, f'Avg: {avg_score:.2f}\nFull: {full_agree}/10',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['grid'], alpha=0.9))

    ax.set_facecolor('#FAFAFA')
    plt.tight_layout()

    return fig


def create_figure7(metrics: Dict[str, GraphMetrics],
                   benchmark_summary: Optional[ExternalValidationSummary] = None) -> plt.Figure:
    """Figure 7: Model vs Benchmark Comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Communication comparison
    ax1 = axes[0]

    if benchmark_summary and benchmark_summary.protocol_results:
        results = benchmark_summary.protocol_results
        model_comm = [r.model_comm_bits / 8 for r in results]  # Convert to bytes
        measured_comm = [r.measured_comm_bytes for r in results]
        labels = [r.protocol_id for r in results]

        ax1.scatter(model_comm, measured_comm, c=COLORS['primary'], s=100,
                    edgecolors='white', linewidths=1.5, alpha=0.85, zorder=3)

        # Add labels
        for i, label in enumerate(labels):
            ax1.annotate(label, (model_comm[i], measured_comm[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=9, color=COLORS['text'])

        # Diagonal line
        max_val = max(max(model_comm), max(measured_comm)) * 1.1
        ax1.plot([0, max_val], [0, max_val], '--', color='#888888',
                 linewidth=1.5, alpha=0.6, zorder=1)

        ax1.set_xlim(0, max_val)
        ax1.set_ylim(0, max_val)

        # Correlation label
        ax1.text(0.05, 0.95, f'ρ = {benchmark_summary.comm_spearman_rho:.3f}',
                 transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No benchmark data', transform=ax1.transAxes,
                 ha='center', va='center', fontsize=12, color='#888888')

    ax1.set_xlabel('Model (bytes)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Measured (bytes)', fontsize=11, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='-', alpha=0.15, color=COLORS['grid'])

    for spine in ax1.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(0.8)

    ax1.text(0.5, -0.12, '(a)', transform=ax1.transAxes,
             fontsize=11, fontweight='bold', ha='center')

    # Right: Computation comparison
    ax2 = axes[1]

    if benchmark_summary and benchmark_summary.protocol_results:
        results = benchmark_summary.protocol_results
        model_comp = [r.model_comp_aes for r in results]
        measured_comp = [r.measured_comp_ms for r in results]

        ax2.scatter(model_comp, measured_comp, c=COLORS['secondary'], s=100,
                    edgecolors='white', linewidths=1.5, alpha=0.85, zorder=3)

        for i, label in enumerate(labels):
            ax2.annotate(label, (model_comp[i], measured_comp[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=9, color=COLORS['text'])

        ax2.text(0.05, 0.95, f'ρ = {benchmark_summary.comp_spearman_rho:.3f}',
                 transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No benchmark data', transform=ax2.transAxes,
                 ha='center', va='center', fontsize=12, color='#888888')

    ax2.set_xlabel('Model (AES ops)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Measured (ms)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='-', alpha=0.15, color=COLORS['grid'])

    for spine in ax2.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(0.8)

    ax2.text(0.5, -0.12, '(b)', transform=ax2.transAxes,
             fontsize=11, fontweight='bold', ha='center')

    ax1.set_facecolor('#FAFAFA')
    ax2.set_facecolor('#FAFAFA')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)

    return fig


def save_figure(fig: plt.Figure, output_dir: str, filename: str) -> None:
    """Save as SVG and PNG"""
    svg_dir = os.path.join(output_dir, 'svg')
    png_dir = os.path.join(output_dir, 'png')

    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    svg_path = os.path.join(svg_dir, f'{filename}.svg')
    png_path = os.path.join(png_dir, f'{filename}.png')

    fig.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"  Saved: {filename}")


def generate_all_figures(
    protocols: Dict[str, Protocol],
    graphs: Dict[str, CommunicationGraph],
    metrics: Dict[str, GraphMetrics],
    selection_results: List[SelectionResult],
    output_dir: str,
    benchmark_summary: Optional[ExternalValidationSummary] = None
) -> None:
    """Generate and save all figures"""
    print("\nGenerating figures...")

    fig1 = create_figure1(protocols, graphs)
    save_figure(fig1, output_dir, 'Figure1_Temporal_Communication_Patterns')
    plt.close(fig1)

    fig2 = create_figure2(graphs)
    save_figure(fig2, output_dir, 'Figure2_Communication_Graph_Structures')
    plt.close(fig2)

    fig3 = create_figure3(metrics)
    save_figure(fig3, output_dir, 'Figure3_Radar_Chart_Comparison')
    plt.close(fig3)

    fig4 = create_figure4(metrics, protocols)
    save_figure(fig4, output_dir, 'Figure4_Protocol_Distribution_Scatter')
    plt.close(fig4)

    fig5 = create_figure5()
    save_figure(fig5, output_dir, 'Figure5_Information_Preservation')
    plt.close(fig5)

    fig6 = create_figure6(selection_results)
    save_figure(fig6, output_dir, 'Figure6_Protocol_Selection_Accuracy')
    plt.close(fig6)

    fig7 = create_figure7(metrics, benchmark_summary)
    save_figure(fig7, output_dir, 'Figure7_Model_Benchmark_Comparison')
    plt.close(fig7)

    print("All figures generated!")


if __name__ == "__main__":
    from protocols import get_protocols, SecurityParameters
    from graph_model import GraphModelBuilder
    from metrics import MetricsExtractor
    from validation import SelectionValidator

    # Create test data
    params = SecurityParameters(security_level=128, field_size=256)
    protocols = get_protocols(params)
    graphs = GraphModelBuilder.build_all_graphs(protocols)
    metrics = MetricsExtractor.extract_all_metrics(graphs)
    selection_results = SelectionValidator.validate_selection_accuracy(metrics, protocols)

    # Generate figures
    output_dir = r'D:\systemfiles\OLE-Nancy\2.6-code\output\figures'
    generate_all_figures(protocols, graphs, metrics, selection_results, output_dir)
