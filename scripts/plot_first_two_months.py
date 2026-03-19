#!/usr/bin/env python3
"""
绘制前两个月策略变动图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def plot_first_two_months(episode_dir, output_file=None):
    """绘制前两个月策略和benchmark的净值曲线"""
    json_path = os.path.join(episode_dir, 'backtest_results.json')
    
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return False
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    portfolio_values = data.get('portfolio_values', [])
    benchmark_curves = data.get('benchmark_curves', {})
    episode = os.path.basename(episode_dir)
    
    if len(portfolio_values) == 0:
        print(f"错误: {episode_dir} 中没有投资组合数据")
        return False
    
    # 前两个月约42个交易日
    days = min(42, len(portfolio_values))
    
    # 策略数据
    strategy_pv = portfolio_values[:days]
    strategy_returns = [(pv / strategy_pv[0] - 1) * 100 for pv in strategy_pv]
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 图1: 净值曲线
    ax1 = axes[0]
    x = range(days)
    
    # 策略曲线
    ax1.plot(x, strategy_returns, 'b-', linewidth=2, label='RL Strategy', marker='o', markersize=3)
    
    # Benchmark 曲线
    colors = {'Buy & Hold': 'r', 'Best Strategy': 'g', 'Random Strategy': 'orange', 
              'Fixed Rotation': 'purple', 'Majority Voting': 'brown', 'Strategy Ensemble': 'pink'}
    
    for name, curve_data in benchmark_curves.items():
        returns = curve_data.get('cumulative_returns', [])[:days]
        if len(returns) == days:
            bm_returns = [r * 100 for r in returns]
            color = colors.get(name, 'gray')
            ax1.plot(x, bm_returns, '--', linewidth=1.5, label=name, alpha=0.7)
    
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.set_xlabel('Trading Days', fontsize=12)
    ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax1.set_title(f'{episode} - First 2 Months Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 添加月度分隔线
    if days >= 21:
        ax1.axvline(x=21, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax1.text(10, ax1.get_ylim()[1]*0.9, 'Month 1', fontsize=10, ha='center', alpha=0.7)
        ax1.text(31, ax1.get_ylim()[1]*0.9, 'Month 2', fontsize=10, ha='center', alpha=0.7)
    
    # 图2: 日收益率柱状图
    ax2 = axes[1]
    daily_returns = []
    for i in range(1, len(strategy_pv)):
        daily_ret = (strategy_pv[i] / strategy_pv[i-1] - 1) * 100
        daily_returns.append(daily_ret)
    
    colors_bar = ['green' if r > 0 else 'red' for r in daily_returns]
    ax2.bar(range(1, len(daily_returns)+1), daily_returns, color=colors_bar, alpha=0.6, width=0.8)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    if days >= 21:
        ax2.axvline(x=21, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Trading Days', fontsize=12)
    ax2.set_ylabel('Daily Return (%)', fontsize=12)
    ax2.set_title('Strategy Daily Returns', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    total_return = strategy_returns[-1]
    max_return = max(strategy_returns)
    min_return = min(strategy_returns)
    
    stats_text = f'Total Return: {total_return:+.2f}%\nMax: {max_return:+.2f}%\nMin: {min_return:+.2f}%'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图表
    if output_file is None:
        output_file = os.path.join(episode_dir, 'first_two_months_chart.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_file}")
    
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description='绘制前两个月策略变动图')
    parser.add_argument('--backtest-dir', type=str, default='backtest_v2',
                        help='回测结果根目录 (默认: backtest_v2)')
    parser.add_argument('--episodes', type=str, default=None,
                        help='指定 episodes，逗号分隔 (例如: 10,20,30)，默认处理所有')
    
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
    
    print(f"=" * 80)
    print(f"绘制前两个月策略变动图")
    print(f"回测目录: {base_dir}")
    print(f"Episodes: {episodes}")
    print(f"=" * 80)
    
    success_count = 0
    for ep in episodes:
        episode_dir = os.path.join(base_dir, ep)
        print(f"\n处理 {ep}...")
        if plot_first_two_months(episode_dir):
            success_count += 1
    
    print(f"\n" + "=" * 80)
    print(f"完成! 成功生成 {success_count}/{len(episodes)} 个图表")
    print(f"=" * 80)


if __name__ == '__main__':
    main()
