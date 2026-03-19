#!/usr/bin/env python3
"""
多时间段 Metrics 计算脚本
为每个 episode 计算 1个月、2个月...到全年的策略和 benchmark metrics
"""

import json
import numpy as np
import os
import argparse
from datetime import datetime, timedelta


def calculate_metrics_for_period(portfolio_values, days):
    """计算指定天数的 metrics"""
    if len(portfolio_values) < 2:
        return None
    
    actual_days = min(days, len(portfolio_values))
    pv = portfolio_values[:actual_days]
    
    # 总收益率
    total_return = (pv[-1] / pv[0]) - 1
    
    # 日收益率
    daily_returns = []
    for i in range(1, len(pv)):
        daily_return = (pv[i] / pv[i-1]) - 1
        daily_returns.append(daily_return)
    daily_returns = np.array(daily_returns)
    
    # 年化夏普
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0
    
    # 最大回撤
    peak = pv[0]
    max_dd = 0
    for value in pv:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_dd:
            max_dd = drawdown
    
    # 年化收益率
    if actual_days > 0:
        annualized_return = (1 + total_return) ** (252 / actual_days) - 1
    else:
        annualized_return = 0
    
    return {
        'days': actual_days,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'initial': pv[0],
        'final': pv[-1]
    }


def calculate_benchmark_metrics_for_period(benchmark_curves, days):
    """计算 benchmark 指定天数的 metrics"""
    results = {}
    for name, curve_data in benchmark_curves.items():
        returns = curve_data.get('cumulative_returns', [])[:days]
        if len(returns) < 2:
            continue
        
        actual_days = len(returns)
        total_return = returns[-1]
        
        # 日收益率
        daily_returns = []
        for i in range(1, len(returns)):
            daily_return = (1 + returns[i]) / (1 + returns[i-1]) - 1
            daily_returns.append(daily_return)
        daily_returns = np.array(daily_returns)
        
        # 年化夏普
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # 最大回撤
        peak_val = 1.0
        max_dd = 0
        for r in returns:
            val = 1 + r
            if val > peak_val:
                peak_val = val
            drawdown = (peak_val - val) / peak_val
            if drawdown > max_dd:
                max_dd = drawdown
        
        # 年化收益率
        if actual_days > 0:
            annualized_return = (1 + total_return) ** (252 / actual_days) - 1
        else:
            annualized_return = 0
        
        results[name] = {
            'days': actual_days,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd
        }
    
    return results


def generate_period_report(episode_dir, output_file):
    """生成多时间段报告"""
    json_path = os.path.join(episode_dir, 'backtest_results.json')
    
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return False
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    portfolio_values = data.get('portfolio_values', [])
    benchmark_curves = data.get('benchmark_curves', {})
    
    if len(portfolio_values) == 0:
        print(f"错误: {episode_dir} 中没有投资组合数据")
        return False
    
    total_days = len(portfolio_values)
    
    # 定义时间段 (交易日)
    periods = [
        (21, "1个月"),
        (42, "2个月"),
        (63, "3个月"),
        (84, "4个月"),
        (105, "5个月"),
        (126, "6个月"),
        (147, "7个月"),
        (168, "8个月"),
        (189, "9个月"),
        (210, "10个月"),
        (231, "11个月"),
        (total_days, f"全年({total_days}天)")
    ]
    
    # 生成报告
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append(f"多时间段 Metrics 分析报告")
    report_lines.append(f"Episode: {os.path.basename(episode_dir)}")
    report_lines.append(f"总交易日: {total_days}")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 100)
    
    # 策略各时间段表现
    report_lines.append("\n" + "=" * 100)
    report_lines.append("【策略各时间段表现】")
    report_lines.append("=" * 100)
    report_lines.append(f"{'时间段':<12} {'天数':>6} {'总收益率':>12} {'年化收益':>12} {'夏普比率':>10} {'最大回撤':>10}")
    report_lines.append("-" * 100)
    
    strategy_results = {}
    for days, label in periods:
        if days > total_days:
            continue
        metrics = calculate_metrics_for_period(portfolio_values, days)
        if metrics:
            strategy_results[label] = metrics
            report_lines.append(
                f"{label:<12} {metrics['days']:>6} "
                f"{metrics['total_return']*100:>+11.2f}% "
                f"{metrics['annualized_return']*100:>+11.2f}% "
                f"{metrics['sharpe']:>10.4f} "
                f"{metrics['max_drawdown']*100:>9.2f}%"
            )
    
    # Benchmarks 各时间段表现
    report_lines.append("\n" + "=" * 100)
    report_lines.append("【Benchmarks 各时间段表现】")
    report_lines.append("=" * 100)
    
    benchmark_names = ['Buy & Hold', 'Best Strategy', 'Random Strategy', 
                       'Fixed Rotation', 'Majority Voting', 'Strategy Ensemble']
    
    for days, label in periods:
        if days > total_days:
            continue
        
        report_lines.append(f"\n--- {label} ({days}天) ---")
        report_lines.append(f"{'Benchmark':<20} {'总收益':>10} {'年化收益':>10} {'夏普':>8} {'回撤':>8}")
        report_lines.append("-" * 70)
        
        bm_metrics = calculate_benchmark_metrics_for_period(benchmark_curves, days)
        for name in benchmark_names:
            if name in bm_metrics:
                bm = bm_metrics[name]
                report_lines.append(
                    f"{name:<20} {bm['total_return']*100:>+9.2f}% "
                    f"{bm['annualized_return']*100:>+9.2f}% "
                    f"{bm['sharpe']:>8.4f} "
                    f"{bm['max_drawdown']*100:>7.2f}%"
                )
    
    # 策略 vs Buy & Hold 对比
    report_lines.append("\n" + "=" * 100)
    report_lines.append("【策略 vs Buy & Hold 超额收益】")
    report_lines.append("=" * 100)
    report_lines.append(f"{'时间段':<12} {'策略收益':>12} {'B&H收益':>12} {'超额收益':>12} {'胜率':>8}")
    report_lines.append("-" * 100)
    
    for days, label in periods:
        if days > total_days:
            continue
        
        strategy_metrics = calculate_metrics_for_period(portfolio_values, days)
        bm_metrics = calculate_benchmark_metrics_for_period(benchmark_curves, days)
        
        if strategy_metrics and 'Buy & Hold' in bm_metrics:
            strategy_ret = strategy_metrics['total_return']
            bh_ret = bm_metrics['Buy & Hold']['total_return']
            excess = strategy_ret - bh_ret
            win = "✓ 胜" if excess > 0 else "✗ 负"
            report_lines.append(
                f"{label:<12} {strategy_ret*100:>+11.2f}% {bh_ret*100:>+11.2f}% "
                f"{excess*100:>+11.2f}% {win:>8}"
            )
    
    # 保存报告
    report_text = "\n".join(report_lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ 报告已保存: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description='计算多时间段 Metrics')
    parser.add_argument('--backtest-dir', type=str, default='backtest_v2',
                        help='回测结果根目录 (默认: backtest_v2)')
    parser.add_argument('--episodes', type=str, default=None,
                        help='指定 episodes，逗号分隔 (例如: 10,20,30)，默认处理所有')
    
    args = parser.parse_args()
    
    base_dir = args.backtest_dir
    
    if args.episodes:
        episodes = [ep.strip() for ep in args.episodes.split(',')]
    else:
        # 自动发现所有 episode 目录
        episodes = []
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                if item.startswith('ep') and os.path.isdir(os.path.join(base_dir, item)):
                    episodes.append(item)
        episodes.sort()
    
    print(f"=" * 80)
    print(f"多时间段 Metrics 计算")
    print(f"回测目录: {base_dir}")
    print(f"Episodes: {episodes}")
    print(f"=" * 80)
    
    success_count = 0
    for ep in episodes:
        episode_dir = os.path.join(base_dir, ep)
        output_file = os.path.join(episode_dir, 'multi_period_metrics.txt')
        
        print(f"\n处理 {ep}...")
        if generate_period_report(episode_dir, output_file):
            success_count += 1
    
    print(f"\n" + "=" * 80)
    print(f"完成! 成功处理 {success_count}/{len(episodes)} 个 episode")
    print(f"=" * 80)


if __name__ == '__main__':
    main()
