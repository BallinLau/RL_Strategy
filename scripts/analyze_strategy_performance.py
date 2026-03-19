"""
分析 RL Strategy 回测结果
- 各固定策略表现
- 最优回看策略（不可实现的理想情况）
- 策略切换分析
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def calculate_metrics(returns, periods_per_year=252):
    """计算策略指标"""
    if len(returns) == 0:
        return {}
    
    total_return = returns[-1] - 1 if returns[-1] > 0 else returns[-1]
    
    # 计算日收益率
    daily_returns = np.diff(returns) / returns[:-1]
    
    # 年化收益率
    n_years = len(returns) / periods_per_year
    annualized_return = (returns[-1] / returns[0]) ** (1/n_years) - 1 if n_years > 0 else 0
    
    # 波动率
    volatility = np.std(daily_returns) * np.sqrt(periods_per_year)
    
    # 最大回撤
    peak = np.maximum.accumulate(returns)
    drawdown = (peak - returns) / peak
    max_drawdown = np.max(drawdown)
    
    # 夏普比率
    sharpe = annualized_return / volatility if volatility > 0 else 0
    
    # Calmar 比率
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar
    }


def calculate_lookback_best_strategy(benchmark_curves, lookback_days=5):
    """
    计算最优回看策略（不可实现的理想情况）
    每天回看过去 lookback_days 天，选择表现最好的策略
    """
    # 获取所有策略的收益率曲线 (排除 Best Strategy，因为它是作弊的)
    strategy_names = [name for name in benchmark_curves.keys() if name != 'Best Strategy']
    
    # 构建收益率矩阵
    returns_matrix = {}
    min_length = float('inf')
    
    for name in strategy_names:
        curve = benchmark_curves[name]['cumulative_returns']
        # 转换为价格曲线 (cumulative_returns 是相对初始值的收益率)
        prices = np.array(curve) + 1.0
        returns_matrix[name] = prices
        min_length = min(min_length, len(prices))
    
    # 截断到相同长度
    for name in strategy_names:
        returns_matrix[name] = returns_matrix[name][:min_length]
    
    # 计算最优回看策略
    lookback_returns = [1.0]  # 初始值
    selected_strategies = []
    
    for t in range(1, min_length):
        if t <= lookback_days:
            # 前期选择 Buy & Hold
            best_strategy = 'Buy & Hold'
        else:
            # 回看过去 lookback_days 天，选择累计收益最高的策略
            best_return = -float('inf')
            best_strategy = strategy_names[0]
            
            for name in strategy_names:
                # 计算回看期内的收益
                if returns_matrix[name][t - lookback_days] > 0:
                    past_return = returns_matrix[name][t] / returns_matrix[name][t - lookback_days]
                    if past_return > best_return:
                        best_return = past_return
                        best_strategy = name
        
        selected_strategies.append(best_strategy)
        
        # 使用选中的策略当天的收益
        if returns_matrix[best_strategy][t-1] > 0:
            daily_return = returns_matrix[best_strategy][t] / returns_matrix[best_strategy][t-1]
            lookback_returns.append(lookback_returns[-1] * daily_return)
        else:
            lookback_returns.append(lookback_returns[-1])
    
    return np.array(lookback_returns), selected_strategies


def analyze_backtest_result(json_path):
    """分析单个回测结果文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"分析文件: {json_path}")
    print(f"{'='*80}")
    
    # 1. RL Agent 表现
    print("\n📊 RL Agent 表现:")
    print(f"  总收益率: {data['metrics']['total_return']*100:.2f}%")
    print(f"  年化收益率: {data['metrics']['annualized_return']*100:.2f}%")
    print(f"  夏普比率: {data['metrics']['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {data['metrics']['max_drawdown']*100:.2f}%")
    print(f"  Calmar比率: {data['metrics']['calmar_ratio']:.3f}")
    print(f"  策略切换次数: {data['metrics']['strategy_switches']}")
    
    # 2. 各固定策略表现
    print("\n📈 固定策略表现对比:")
    print("-" * 80)
    print(f"{'策略名称':<20} {'总收益':>10} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'Calmar':>8}")
    print("-" * 80)
    
    benchmark_curves = data.get('benchmark_curves', {})
    strategy_metrics = {}
    
    for name, curve_data in benchmark_curves.items():
        returns = np.array(curve_data['cumulative_returns']) + 1
        metrics = calculate_metrics(returns)
        strategy_metrics[name] = metrics
        
        print(f"{name:<20} {metrics['total_return']*100:>9.2f}% {metrics['annualized_return']*100:>9.2f}% "
              f"{metrics['sharpe_ratio']:>8.3f} {metrics['max_drawdown']*100:>9.2f}% {metrics['calmar_ratio']:>8.3f}")
    
    # 3. 最优回看策略（不可实现）
    print("\n🏆 最优回看策略 Lookback-Best (不可实现的理想情况):")
    print("-" * 80)
    
    for lookback in [1, 5, 10, 20]:
        lookback_returns, selected = calculate_lookback_best_strategy(benchmark_curves, lookback)
        metrics = calculate_metrics(lookback_returns)
        
        print(f"\n  回看天数: {lookback}")
        print(f"    总收益率: {metrics['total_return']*100:.2f}%")
        print(f"    年化收益率: {metrics['annualized_return']*100:.2f}%")
        print(f"    夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"    最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"    Calmar比率: {metrics['calmar_ratio']:.3f}")
    
    # 4. 策略使用统计
    print("\n📊 RL Agent 策略选择统计:")
    actions = data.get('actions_history', [])
    strategy_names = data.get('strategy_names_history', [])
    
    if actions and strategy_names:
        unique_actions = sorted(set(actions))
        print(f"\n  共切换策略 {len(actions)} 次:")
        for action in unique_actions:
            count = actions.count(action)
            pct = count / len(actions) * 100
            strategy_name = strategy_names[action] if action < len(strategy_names) else f"Action {action}"
            print(f"    {strategy_name}: {count} 次 ({pct:.1f}%)")
    
    return data, strategy_metrics


def compare_episodes(base_path, episode_range=range(10, 101, 10)):
    """比较多个 episode 的结果"""
    print(f"\n{'='*80}")
    print("多 Episode 对比分析")
    print(f"{'='*80}")
    
    results = []
    
    for ep in episode_range:
        json_path = base_path / f"ep{ep}" / "backtest_results.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            results.append({
                'episode': ep,
                'rl_return': data['metrics']['total_return'],
                'rl_sharpe': data['metrics']['sharpe_ratio'],
                'rl_drawdown': data['metrics']['max_drawdown'],
                'buy_hold_return': data['benchmark_curves']['Buy & Hold']['cumulative_returns'][-1],
                'best_strategy_return': data['benchmark_curves']['Best Strategy']['cumulative_returns'][-1]
            })
    
    if results:
        df = pd.DataFrame(results)
        print("\n📊 各 Episode 表现对比:")
        print("-" * 80)
        print(f"{'Episode':>8} {'RL收益':>10} {'RL夏普':>8} {'RL回撤':>8} {'Buy&Hold':>10} {'Best策略':>10}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"{int(row['episode']):>8} {row['rl_return']*100:>9.2f}% {row['rl_sharpe']:>8.3f} "
                  f"{row['rl_drawdown']*100:>7.2f}% {row['buy_hold_return']*100:>9.2f}% {row['best_strategy_return']*100:>9.2f}%")
        
        # 找出最佳 episode
        best_episode = df.loc[df['rl_sharpe'].idxmax()]
        print(f"\n✅ 最佳 Episode: {int(best_episode['episode'])}")
        print(f"   夏普比率: {best_episode['rl_sharpe']:.3f}")
        print(f"   总收益: {best_episode['rl_return']*100:.2f}%")


def main():
    # 分析最新的 backtest_v2 ep100
    base_path = Path("/home/fit/zhuyingz/WORK/LiuHao/RL_Strategy/backtest_v2")
    
    # 分析单个文件
    json_path = base_path / "ep100" / "backtest_results.json"
    if json_path.exists():
        analyze_backtest_result(json_path)
    
    # 对比多个 episodes
    compare_episodes(base_path)
    
    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)


if __name__ == "__main__":
    main()
