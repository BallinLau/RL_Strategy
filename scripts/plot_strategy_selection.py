#!/usr/bin/env python3
"""
Plot Strategy Selection - Show daily action and strategy choices
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import argparse
from collections import Counter


def plot_strategy_selection(episode_dir, output_file=None):
    """Plot first two months strategy selection"""
    json_path = os.path.join(episode_dir, 'backtest_results.json')
    
    if not os.path.exists(json_path):
        print(f"Error: File not found {json_path}")
        return False
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    actions = data.get('actions_history', [])
    strategies = data.get('strategy_names_history', [])
    portfolio_values = data.get('portfolio_values', [])
    episode = os.path.basename(episode_dir)
    
    if len(actions) == 0:
        print(f"Error: No strategy data in {episode_dir}")
        return False
    
    # First two months ~42 trading days
    days = min(42, len(actions))
    actions = actions[:days]
    strategies = strategies[:days]
    portfolio_values = portfolio_values[:days]
    
    # Get unique strategies and actions
    unique_strategies = sorted(set(strategies))
    unique_actions = sorted(set(actions))
    
    # Create color mapping
    strategy_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_strategies)))
    strategy_color_map = {s: strategy_colors[i] for i, s in enumerate(unique_strategies)}
    
    action_colors = plt.cm.Paired(np.linspace(0, 1, len(unique_actions)))
    action_color_map = {a: action_colors[i] for i, a in enumerate(unique_actions)}
    
    # Create figure with 2 rows
    fig = plt.figure(figsize=(16, 10))
    
    # Row 1: Strategy and Action heatmaps
    gs_top = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, top=0.92, bottom=0.55)
    
    # Subplot 1: Strategy selection
    ax1 = fig.add_subplot(gs_top[0, 0])
    for i, strategy in enumerate(strategies):
        ax1.barh(0, 1, left=i, color=strategy_color_map[strategy], height=1, edgecolor='white', linewidth=0.5)
    
    ax1.set_xlim(0, days)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.set_xlabel('Trading Days', fontsize=10)
    ax1.set_title('Strategy Selection', fontsize=12, fontweight='bold')
    
    legend_patches = [mpatches.Patch(color=strategy_color_map[s], label=s) for s in unique_strategies]
    ax1.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2)
    
    if days >= 21:
        ax1.axvline(x=21, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.text(10, 0.7, 'M1', fontsize=9, ha='center', fontweight='bold')
        ax1.text(31, 0.7, 'M2', fontsize=9, ha='center', fontweight='bold')
    
    # Subplot 2: Action selection
    ax2 = fig.add_subplot(gs_top[1, 0])
    for i, action in enumerate(actions):
        ax2.barh(0, 1, left=i, color=action_color_map[action], height=1, edgecolor='white', linewidth=0.5)
    
    ax2.set_xlim(0, days)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.set_xlabel('Trading Days', fontsize=10)
    ax2.set_title('Action Selection', fontsize=12, fontweight='bold')
    
    legend_patches2 = [mpatches.Patch(color=action_color_map[a], label=f'Act {a}') for a in unique_actions]
    ax2.legend(handles=legend_patches2, loc='upper left', fontsize=8, ncol=2)
    
    if days >= 21:
        ax2.axvline(x=21, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Subplot 3: Strategy duration timeline
    ax3 = fig.add_subplot(gs_top[0, 1])
    
    strategy_changes = [0]
    for i in range(1, len(strategies)):
        if strategies[i] != strategies[i-1]:
            strategy_changes.append(i)
    
    y_pos = 0
    for i in range(len(strategy_changes)):
        start = strategy_changes[i]
        end = strategy_changes[i+1] if i+1 < len(strategy_changes) else len(strategies)
        duration = end - start
        strategy = strategies[start]
        
        ax3.barh(y_pos, duration, left=start, color=strategy_color_map[strategy], 
                height=0.6, edgecolor='white', linewidth=1)
        if duration >= 3:
            ax3.text(start + duration/2, y_pos, strategy[:8], 
                    ha='center', va='center', fontsize=7, fontweight='bold')
        y_pos += 1
    
    ax3.set_xlim(0, days)
    ax3.set_xlabel('Trading Days', fontsize=10)
    ax3.set_ylabel('Switch #', fontsize=10)
    ax3.set_title('Strategy Duration', fontsize=12, fontweight='bold')
    
    if days >= 21:
        ax3.axvline(x=21, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Subplot 4: Portfolio value
    ax4 = fig.add_subplot(gs_top[1, 1])
    
    returns = [(pv / portfolio_values[0] - 1) * 100 for pv in portfolio_values]
    ax4.plot(range(days), returns, 'b-', linewidth=2, marker='o', markersize=2)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax4.fill_between(range(days), returns, 0, alpha=0.3, color='blue')
    
    ax4.set_xlabel('Trading Days', fontsize=10)
    ax4.set_ylabel('Return (%)', fontsize=10)
    ax4.set_title('Cumulative Return', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    if days >= 21:
        ax4.axvline(x=21, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Row 2: Statistics
    gs_bottom = fig.add_gridspec(1, 2, hspace=0.3, top=0.45, bottom=0.08)
    ax5 = fig.add_subplot(gs_bottom[0, 0])
    ax5.axis('off')
    
    strategy_counts = Counter(strategies)
    action_counts = Counter(actions)
    
    stats_text = f"Strategy Statistics - First 2 Months\n\n"
    stats_text += f"Strategy Distribution:\n"
    for strategy, count in strategy_counts.most_common():
        stats_text += f"  {strategy}: {count} days ({count/days*100:.1f}%)\n"
    
    stats_text += f"\nAction Distribution:\n"
    for action, count in action_counts.most_common():
        stats_text += f"  Action {action}: {count} days ({count/days*100:.1f}%)\n"
    
    stats_text += f"\nTotal Switches: {len(strategy_changes)-1}\n"
    stats_text += f"Avg Hold: {days/len(strategy_changes):.1f} days\n"
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Timeline details
    ax6 = fig.add_subplot(gs_bottom[0, 1])
    ax6.axis('off')
    
    timeline_text = f"Strategy Timeline:\n\n"
    for i in range(min(8, len(strategy_changes))):  # Show first 8 switches
        start = strategy_changes[i]
        end = strategy_changes[i+1] if i+1 < len(strategy_changes) else len(strategies)
        duration = end - start
        strategy = strategies[start]
        timeline_text += f"Days {start+1:2d}-{end:2d}: {strategy:15s} ({duration:2d} days)\n"
    
    if len(strategy_changes) > 8:
        timeline_text += f"... and {len(strategy_changes)-8} more switches\n"
    
    ax6.text(0.05, 0.95, timeline_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle(f'{episode} - Strategy Selection Analysis', fontsize=14, fontweight='bold', y=0.98)
    
    # Save chart
    if output_file is None:
        output_file = os.path.join(episode_dir, 'strategy_selection_chart.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Chart saved: {output_file}")
    
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description='Plot strategy selection')
    parser.add_argument('--backtest-dir', type=str, default='backtest_v2',
                        help='Backtest root directory (default: backtest_v2)')
    parser.add_argument('--episodes', type=str, default=None,
                        help='Episodes to plot, comma separated (e.g., 10,20,30)')
    
    args = parser.parse_args()
    
    base_dir = args.backtest_dir
    
    if args.episodes:
        episodes = [f"ep{ep.strip()}" for ep in args.episodes.split(',')]
    else:
        episodes = []
        if os.path.exists(base_dir):
            for item in sorted(os.listdir(base_dir)):
                if item.startswith('ep') and os.path.isdir(os.path.join(base_dir, item)):
                    episodes.append(item)
    
    print(f"Plotting strategy selection charts")
    print(f"Directory: {base_dir}")
    print(f"Episodes: {episodes}")
    
    success_count = 0
    for ep in episodes:
        episode_dir = os.path.join(base_dir, ep)
        print(f"Processing {ep}...")
        if plot_strategy_selection(episode_dir):
            success_count += 1
    
    print(f"Done! Generated {success_count}/{len(episodes)} charts")


if __name__ == '__main__':
    main()
