#!/usr/bin/env python3
"""
按月计算 Metrics 脚本
为每个 episode 计算每个月的独立 metrics
"""

import json
import numpy as np
import os
import argparse
from datetime import datetime, timedelta


def calculate_metrics_for_period(portfolio_values, start_idx, end_idx):
    """计算指定索引范围的 metrics"""
    if end_idx <= start_idx or end_idx > len(portfolio_values):
        return None
    
    pv = portfolio_values[start_idx:end_idx]
    days = len(pv)
    
    if days < 2:
        return None
    
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
    
    return {
        'days': days,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'initial': pv[0],
        'final': pv[-1]
    }


def calculate_benchmark_metrics_for_period(benchmark_curves, start_idx, end_idx):
    """计算 benchmark 指定索引范围的 metrics"""
    results = {}
    for name, curve_data in benchmark_curves.items():
        returns = curve_data.get('cumulative_returns', [])[start_idx:end_idx]
        if len(returns) < 2:
            continue
        
        days = len(returns)
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
        
        results[name] = {
            'days': days,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd
        }
    
    return results


def generate_monthly_report(episode_dir, output_file):
    """生成按月报告"""
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
    
    # 假设每月约21个交易日
    trading_days_per_month = 21
    
    # 计算有多少个完整的月
    num_months = (total_days + trading_days_per_month - 1) // trading_days_per_month
    
    # 生成报告
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append(f"按月 Metrics 分析报告")
    report_lines.append(f"Episode: {os.path.basename(episode_dir)}")
    report_lines.append(f"总交易日: {total_days}")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 100)
    
    # 策略各月表现
    report_lines.append("\n" + "=" * 100)
    report_lines.append("【策略各月表现】")
    report_lines.append("=" * 100)
    report_lines.append(f"{'月份':<10} {'天数':>6} {'起始净值':>14} {'结束净值':>14} {'月收益率':>12} {'夏普比率':>10} {'最大回撤':>10}")
    report_lines.append("-" * 100)
    
    monthly_results = {}
    for month in range(1, num_months + 1):
        start_idx = (month - 1) * trading_days_per_month
        end_idx = min(month * trading_days_per_month, total_days)
        
        metrics = calculate_metrics_for_period(portfolio_values, start_idx, end_idx)
        if metrics:
            monthly_results[f"第{month}月"] = metrics
            report_lines.append(
                f"第{month}月{'':<6} {metrics['days']:>6} "
                f"{metrics['initial']:>14,.0f} {metrics['final']:>14,.0f} "
                f"{metrics['total_return']*100:>+11.2f}% {metrics['sharpe']:>10.4f} {metrics['max_drawdown']*100:>9.2f}%"
            )
    
    # Benchmarks 各月表现
    report_lines.append("\n" + "=" * 100)
    report_lines.append("【Benchmarks 各月表现】")
    report_lines.append("=" * 100)
    
    benchmark_names = ['Buy & Hold', 'Best Strategy', 'Random Strategy', 
                       'Fixed Rotation', 'Majority Voting', 'Strategy Ensemble']
    
    for month in range(1, num_months + 1):
        start_idx = (month - 1) * trading_days_per_month
        end_idx = min(month * trading_days_per_month, total_days)
        
        report_lines.append(f"\n--- 第{month}月 ---")
        report_lines.append(f"{'Benchmark':<20} {'月收益':>10} {'夏普':>8} {'回撤':>8}")
        report_lines.append("-" * 60)
        
        bm_metrics = calculate_benchmark_metrics_for_period(benchmark_curves, start_idx, end_idx)
        for name in benchmark_names:
            if name in bm_metrics:
                bm = bm_metrics[name]
                report_lines.append(
                    f"{name:<20} {bm['total_return']*100:>+9.2f}% {bm['sharpe']:>8.4f} {bm['max_drawdown']*100:>7.2f}%"
                )
    
    # 策略 vs Buy & Hold 每月对比
    report_lines.append("\n" + "=" * 100)
    report_lines.append("【策略 vs Buy & Hold 每月对比】")
    report_lines.append("=" * 100)
    report_lines.append(f"{'月份':<10} {'策略收益':>12} {'B&H收益':>12} {'超额收益':>12} {'胜率':>8}")
    report_lines.append("-" * 100)
    
    for month in range(1, num_months + 1):
        start_idx = (month - 1) * trading_days_per_month
        end_idx = min(month * trading_days_per_month, total_days)
        
        strategy_metrics = calculate_metrics_for_period(portfolio_values, start_idx, end_idx)
        bm_metrics = calculate_benchmark_metrics_for_period(benchmark_curves, start_idx, end_idx)
        
        if strategy_metrics and 'Buy & Hold' in bm_metrics:
            strategy_ret = strategy_metrics['total_return']
            bh_ret = bm_metrics['Buy & Hold']['total_return']
            excess = strategy_ret - bh_ret
            win = "✓ 胜" if excess > 0 else "✗ 负"
            report_lines.append(
                f"第{month}月{'':<6} {strategy_ret*100:>+11.2f}% {bh_ret*100:>+11.2f}% "
                f"{excess*100:>+11.2f}% {win:>8}"
            )
    
    # 月度统计
    report_lines.append("\n" + "=" * 100)
    report_lines.append("【月度统计汇总】")
    report_lines.append("=" * 100)
    
    positive_months = sum(1 for m in monthly_results.values() if m['total_return'] > 0)
    negative_months = len(monthly_results) - positive_months
    avg_monthly_return = np.mean([m['total_return'] for m in monthly_results.values()]) if monthly_results else 0
    best_month = max(monthly_results.items(), key=lambda x: x[1]['total_return']) if monthly_results else (None, None)
    worst_month = min(monthly_results.items(), key=lambda x: x[1]['total_return']) if monthly_results else (None, None)
    
    report_lines.append(f"盈利月数: {positive_months}")
    report_lines.append(f"亏损月数: {negative_months}")
    report_lines.append(f"胜率: {positive_months/len(monthly_results)*100:.1f}%" if monthly_results else "胜率: N/A")
    report_lines.append(f"平均月收益率: {avg_monthly_return*100:+.2f}%")
    if best_month[0]:
        report_lines.append(f"最佳月份: {best_month[0]} ({best_month[1]['total_return']*100:+.2f}%)")
    if worst_month[0]:
        report_lines.append(f"最差月份: {worst_month[0]} ({worst_month[1]['total_return']*100:+.2f}%)")
    
    # 保存报告
    report_text = "\n".join(report_lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ 按月报告已保存: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description='计算按月 Metrics')
    parser.add_argument('--backtest-dir', type=str, default='backtest_v2',
                        help='回测结果根目录 (默认: backtest_v2)')
    parser.add_argument('--episodes', type=str, default=None,
                        help='指定 episodes，逗号分隔 (例如: 10,20,30)，默认处理所有')
    
    args = parser.parse_args()
    
    base_dir = args.backtest_dir
    
    if args.episodes:
        episodes = [f"ep{ep.strip()}" for ep in args.episodes.split(',')]
    else:
        # 自动发现所有 episode 目录
        episodes = []
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                if item.startswith('ep') and os.path.isdir(os.path.join(base_dir, item)):
                    episodes.append(item)
        episodes.sort()
    
    print(f"=" * 80)
    print(f"按月 Metrics 计算")
    print(f"回测目录: {base_dir}")
    print(f"Episodes: {episodes}")
    print(f"=" * 80)
    
    success_count = 0
    for ep in episodes:
        episode_dir = os.path.join(base_dir, ep)
        output_file = os.path.join(episode_dir, 'monthly_metrics.txt')
        
        print(f"\n处理 {ep}...")
        if generate_monthly_report(episode_dir, output_file):
            success_count += 1
    
    print(f"\n" + "=" * 80)
    print(f"完成! 成功处理 {success_count}/{len(episodes)} 个 episode")
    print(f"=" * 80)


if __name__ == '__main__':
    main()
