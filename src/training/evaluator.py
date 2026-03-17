"""
评估器 (Evaluator)

负责模型性能评估和指标计算
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import json

from src.environment.trading_env import TradingEnvironment
from src.agent.ddqn_agent import DDQNAgent


class Evaluator:
    """
    性能评估器
    
    负责：
    1. 性能指标计算
    2. 评估报告生成
    3. 可视化分析
    """
    
    def __init__(self, env: TradingEnvironment, agent: DDQNAgent):
        """
        初始化评估器（增强版）
        
        Args:
            env: 交易环境
            agent: DDQN智能体
        """
        self.env = env
        self.agent = agent
        
        # 评估历史
        self.evaluation_history = []
        
        # 长期评估数据（文档要求）
        self.long_term_metrics = {
            'daily_switch_counts': [],      # 日均切换次数
            'switch_win_rates': [],         # 切换胜率
            'avg_holding_periods': [],      # 平均持仓时长
            'switch_frequency_cv': [],      # 切换频率变异系数
            'market_state_win_rates': {},   # 市场状态下胜率
            'switch_costs': [],             # 切换成本占比
            'intraday_drawdown_freq': []    # 日内回撤超限频率
        }
        
        # 参数调整记录
        self.parameter_adjustments = []
    
    def evaluate(self, 
                 num_episodes: int = 10,
                 render: bool = False,
                 save_results: bool = True,
                 output_dir: str = "outputs") -> Dict[str, Any]:
        """
        全面评估模型性能
        
        Args:
            num_episodes: 评估episode数量
            render: 是否渲染环境
            save_results: 是否保存结果
            output_dir: 输出目录
        
        Returns:
            evaluation_results: 评估结果
        """
        print(f"开始全面评估，{num_episodes} 个episode")
        
        # 收集评估数据
        episode_data = self._collect_episode_data(num_episodes, render)
        
        # 计算性能指标
        performance_metrics = self._calculate_performance_metrics(episode_data)
        
        # 分析交易行为
        trading_analysis = self._analyze_trading_behavior(episode_data)
        
        # 分析策略选择
        strategy_analysis = self._analyze_strategy_selection(episode_data)
        
        # 组合分析
        portfolio_analysis = self._analyze_portfolio_performance(episode_data)
        
        # 综合评估结果
        evaluation_results = {
            'performance_metrics': performance_metrics,
            'trading_analysis': trading_analysis,
            'strategy_analysis': strategy_analysis,
            'portfolio_analysis': portfolio_analysis,
            'episode_count': num_episodes,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # 保存评估历史
        self.evaluation_history.append(evaluation_results)
        
        # 保存结果
        if save_results:
            self._save_evaluation_results(evaluation_results, output_dir)
            self._generate_visualizations(episode_data, evaluation_results, output_dir)
        
        # 打印摘要
        self._print_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    def _collect_episode_data(self, num_episodes: int, render: bool) -> List[Dict[str, Any]]:
        """
        收集episode数据
        
        Args:
            num_episodes: episode数量
            render: 是否渲染
        
        Returns:
            episode_data: 每个episode的数据
        """
        episode_data = []
        
        for episode_idx in range(num_episodes):
            print(f"  运行episode {episode_idx + 1}/{num_episodes}")
            
            # 重置环境
            state = self.env.reset()
            episode_initial_value = float(getattr(self.env, 'initial_capital', 0))
            episode_history = []
            
            done = False
            while not done:
                # 选择动作（评估模式）
                action = self.agent.select_action(state, training=False)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 记录步骤数据
                step_data = {
                    'state': state.copy(),
                    'action': action,
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done,
                    'info': info.copy()
                }
                episode_history.append(step_data)
                
                # 更新状态
                state = next_state
                
                if render:
                    self.env.render()
            
            # 收集episode数据
            episode_summary = {
                'episode_idx': episode_idx,
                'total_reward': sum(step['reward'] for step in episode_history),
                'episode_length': len(episode_history),
                'final_portfolio_value': episode_history[-1]['info'].get('portfolio_value', 0),
                'initial_portfolio_value': episode_initial_value,
                'history': episode_history
            }
            episode_data.append(episode_summary)
        
        return episode_data
    
    def _calculate_performance_metrics(self, episode_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算性能指标
        
        Args:
            episode_data: episode数据
        
        Returns:
            performance_metrics: 性能指标
        """
        # 提取关键数据
        total_rewards = [ep['total_reward'] for ep in episode_data]
        portfolio_values = [ep['final_portfolio_value'] for ep in episode_data]
        initial_values = [ep['initial_portfolio_value'] for ep in episode_data]
        
        # 计算收益指标
        returns = [(pv - iv) / iv for pv, iv in zip(portfolio_values, initial_values)]
        
        # 基本统计
        metrics = {
            # 收益指标
            'total_rewards': {
                'mean': float(np.mean(total_rewards)),
                'std': float(np.std(total_rewards)),
                'min': float(np.min(total_rewards)),
                'max': float(np.max(total_rewards)),
                'median': float(np.median(total_rewards))
            },
            'portfolio_returns': {
                'mean': float(np.mean(returns)),
                'std': float(np.std(returns)),
                'min': float(np.min(returns)),
                'max': float(np.max(returns)),
                'median': float(np.median(returns))
            },
            'portfolio_values': {
                'mean': float(np.mean(portfolio_values)),
                'std': float(np.std(portfolio_values)),
                'min': float(np.min(portfolio_values)),
                'max': float(np.max(portfolio_values)),
                'final': float(portfolio_values[-1]) if portfolio_values else 0
            },
            
            # 风险调整指标
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(portfolio_values),
            
            # 风险控制指标
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'volatility': float(np.std(returns) * np.sqrt(252)),  # 年化波动率
            'win_rate': self._calculate_win_rate(returns),
            
            # 交易行为指标
            'avg_episode_length': float(np.mean([ep['episode_length'] for ep in episode_data])),
            'total_trading_steps': sum([ep['episode_length'] for ep in episode_data])
        }
        
        return metrics
    
    def _analyze_trading_behavior(self, episode_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析交易行为
        
        Args:
            episode_data: episode数据
        
        Returns:
            trading_analysis: 交易行为分析
        """
        all_actions = []
        all_rewards = []
        all_portfolio_changes = []
        prev_portfolio_value = None
        
        for ep in episode_data:
            for step in ep['history']:
                all_actions.append(step['action'])
                all_rewards.append(step['reward'])
                
                # 计算组合价值变化
                if 'portfolio_value' in step['info']:
                    portfolio_value = step['info']['portfolio_value']
                    if prev_portfolio_value is not None and prev_portfolio_value != 0:
                        change = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
                    else:
                        change = 0
                    all_portfolio_changes.append(change)
                    prev_portfolio_value = portfolio_value
        
        # 动作分布
        action_counts = {}
        for action in all_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # 奖励分析
        reward_stats = {
            'mean': float(np.mean(all_rewards)) if all_rewards else 0,
            'std': float(np.std(all_rewards)) if all_rewards else 0,
            'positive_ratio': float(np.mean([r > 0 for r in all_rewards])) if all_rewards else 0
        }
        
        # 交易频率
        total_steps = len(all_actions)
        trading_analysis = {
            'action_distribution': action_counts,
            'reward_statistics': reward_stats,
            'total_trading_steps': total_steps,
            'avg_reward_per_step': reward_stats['mean'],
            'portfolio_change_stats': {
                'mean': float(np.mean(all_portfolio_changes)) if all_portfolio_changes else 0,
                'std': float(np.std(all_portfolio_changes)) if all_portfolio_changes else 0
            }
        }
        
        return trading_analysis
    
    def _analyze_strategy_selection(self, episode_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析策略选择
        
        Args:
            episode_data: episode数据
        
        Returns:
            strategy_analysis: 策略选择分析
        """
        strategy_counts = {}
        strategy_rewards = {}
        strategy_portfolio_changes = {}
        
        for ep in episode_data:
            for step in ep['history']:
                action = step['action']
                strategy_name = step['info'].get('strategy', f'Strategy_{action}')
                
                # 计数
                strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
                
                # 累计奖励
                if strategy_name not in strategy_rewards:
                    strategy_rewards[strategy_name] = []
                strategy_rewards[strategy_name].append(step['reward'])
                
                # 组合价值变化
                if 'portfolio_value' in step['info']:
                    if strategy_name not in strategy_portfolio_changes:
                        strategy_portfolio_changes[strategy_name] = []
                    portfolio_value = step['info']['portfolio_value']
                    strategy_portfolio_changes[strategy_name].append(portfolio_value)
        
        # 计算策略性能
        strategy_performance = {}
        for strategy in strategy_counts.keys():
            rewards = strategy_rewards.get(strategy, [])
            portfolio_values = strategy_portfolio_changes.get(strategy, [])
            
            if rewards:
                avg_reward = np.mean(rewards)
                reward_std = np.std(rewards)
            else:
                avg_reward = 0
                reward_std = 0
            
            if len(portfolio_values) > 1:
                returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                          for i in range(1, len(portfolio_values))]
                avg_return = np.mean(returns) if returns else 0
            else:
                avg_return = 0
            
            strategy_performance[strategy] = {
                'count': strategy_counts[strategy],
                'percentage': strategy_counts[strategy] / sum(strategy_counts.values()) * 100,
                'avg_reward': float(avg_reward),
                'reward_std': float(reward_std),
                'avg_return': float(avg_return)
            }
        
        # 按使用频率排序
        sorted_strategies = sorted(strategy_performance.items(), 
                                  key=lambda x: x[1]['count'], 
                                  reverse=True)
        
        strategy_analysis = {
            'strategy_performance': dict(sorted_strategies),
            'total_strategy_switches': self._count_strategy_switches(episode_data),
            'most_used_strategy': sorted_strategies[0][0] if sorted_strategies else 'N/A',
            'strategy_diversity': len(strategy_counts) / 6  # 6种策略
        }
        
        return strategy_analysis
    
    def _analyze_portfolio_performance(self, episode_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析组合性能
        
        Args:
            episode_data: episode数据
        
        Returns:
            portfolio_analysis: 组合性能分析
        """
        # 提取所有时间点的组合价值
        all_portfolio_values = []
        all_timestamps = []
        
        timestamp = 0
        for ep in episode_data:
            for step in ep['history']:
                if 'portfolio_value' in step['info']:
                    all_portfolio_values.append(step['info']['portfolio_value'])
                    all_timestamps.append(timestamp)
                    timestamp += 1
        
        if not all_portfolio_values:
            return {}
        
        # 计算组合指标
        initial_value = all_portfolio_values[0]
        final_value = all_portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # 计算滚动指标
        window_size = min(20, len(all_portfolio_values) // 10)
        rolling_returns = []
        rolling_volatility = []
        
        for i in range(window_size, len(all_portfolio_values)):
            window_values = all_portfolio_values[i-window_size:i]
            returns = [(window_values[j] - window_values[j-1]) / window_values[j-1] 
                      for j in range(1, len(window_values))]
            rolling_returns.append(np.mean(returns) if returns else 0)
            rolling_volatility.append(np.std(returns) if returns else 0)
        
        portfolio_analysis = {
            'initial_value': float(initial_value),
            'final_value': float(final_value),
            'total_return': float(total_return),
            'annualized_return': float(total_return * 252 / len(all_portfolio_values)) if all_portfolio_values else 0,
            'value_series': all_portfolio_values,
            'timestamps': all_timestamps,
            'rolling_returns': rolling_returns,
            'rolling_volatility': rolling_volatility,
            'value_at_risk': self._calculate_value_at_risk(all_portfolio_values),
            'conditional_var': self._calculate_conditional_var(all_portfolio_values)
        }
        
        return portfolio_analysis
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if not returns or np.std(returns) == 0:
            return 0.0
        
        excess_returns = [r - risk_free_rate/252 for r in returns]  # 日度无风险利率
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        if not returns:
            return 0.0
        
        excess_returns = [r - risk_free_rate/252 for r in returns]
        downside_returns = [r for r in excess_returns if r < 0]
        
        if not downside_returns or np.std(downside_returns) == 0:
            return 0.0
        
        return float(np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252))
    
    def _calculate_calmar_ratio(self, portfolio_values: List[float]) -> float:
        """计算卡玛比率"""
        if not portfolio_values or len(portfolio_values) < 2:
            return 0.0
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        max_dd = self._calculate_max_drawdown(portfolio_values)
        
        if max_dd == 0:
            return 0.0
        
        return float(total_return / max_dd)
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """计算最大回撤"""
        if not portfolio_values:
            return 0.0
        
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return float(max_dd)
    
    def _calculate_win_rate(self, returns: List[float]) -> float:
        """计算胜率"""
        if not returns:
            return 0.0
        
        positive_returns = sum(1 for r in returns if r > 0)
        return float(positive_returns / len(returns))
    
    def _calculate_value_at_risk(self, portfolio_values: List[float], confidence_level: float = 0.95) -> float:
        """计算在险价值"""
        if len(portfolio_values) < 2:
            return 0.0
        
        returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                  for i in range(1, len(portfolio_values))]
        
        if not returns:
            return 0.0
        
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return float(-var * portfolio_values[-1])  # 转换为金额
    
    def _calculate_conditional_var(self, portfolio_values: List[float], confidence_level: float = 0.95) -> float:
        """计算条件在险价值"""
        if len(portfolio_values) < 2:
            return 0.0
        
        returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                  for i in range(1, len(portfolio_values))]
        
        if not returns:
            return 0.0
        
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        tail_returns = [r for r in returns if r <= var_threshold]
        
        if not tail_returns:
            return 0.0
        
        cvar = np.mean(tail_returns)
        return float(-cvar * portfolio_values[-1])
    
    def _count_strategy_switches(self, episode_data: List[Dict[str, Any]]) -> int:
        """计算策略切换次数"""
        total_switches = 0
        
        for ep in episode_data:
            prev_action = None
            for step in ep['history']:
                current_action = step['action']
                if prev_action is not None and current_action != prev_action:
                    total_switches += 1
                prev_action = current_action
        
        return total_switches
    
    def _save_evaluation_results(self, evaluation_results: Dict[str, Any], output_dir: str):
        """保存评估结果"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        
        # 转换numpy类型为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(evaluation_results)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已保存到: {json_path}")
    
    def _generate_visualizations(self, 
                                episode_data: List[Dict[str, Any]], 
                                evaluation_results: Dict[str, Any],
                                output_dir: str):
        """生成可视化图表"""
        import os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 组合价值曲线
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 组合价值时间序列
        portfolio_values = []
        for ep in episode_data:
            for step in ep['history']:
                if 'portfolio_value' in step['info']:
                    portfolio_values.append(step['info']['portfolio_value'])
        
        if portfolio_values:
            axes[0, 0].plot(portfolio_values)
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_xlabel('Trading Step')
            axes[0, 0].set_ylabel('Portfolio Value')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 策略分布
        strategy_analysis = evaluation_results.get('strategy_analysis', {})
        strategy_performance = strategy_analysis.get('strategy_performance', {})
        
        if strategy_performance:
            strategies = list(strategy_performance.keys())
            percentages = [sp['percentage'] for sp in strategy_performance.values()]
            
            axes[0, 1].bar(strategies, percentages)
            axes[0, 1].set_title('Strategy Usage Distribution')
            axes[0, 1].set_xlabel('Strategy')
            axes[0, 1].set_ylabel('Usage Percentage (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 奖励分布
        all_rewards = []
        for ep in episode_data:
            for step in ep['history']:
                all_rewards.append(step['reward'])
        
        if all_rewards:
            axes[1, 0].hist(all_rewards, bins=50, alpha=0.7)
            axes[1, 0].axvline(x=np.mean(all_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(all_rewards):.2f}')
            axes[1, 0].set_title('Reward Distribution')
            axes[1, 0].set_xlabel('Reward')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 性能指标雷达图
        metrics = evaluation_results.get('performance_metrics', {})
        if metrics:
            radar_labels = ['Sharpe', 'Sortino', 'Calmar', 'Win Rate', 'Return']
            radar_values = [
                metrics.get('sharpe_ratio', 0),
                metrics.get('sortino_ratio', 0),
                metrics.get('calmar_ratio', 0),
                metrics.get('win_rate', 0),
                metrics.get('portfolio_returns', {}).get('mean', 0) * 100
            ]
            
            # 归一化到0-1范围
            max_val = max(radar_values) if radar_values else 1
            if max_val > 0:
                radar_values = [v / max_val for v in radar_values]
            
            angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
            radar_values += radar_values[:1]
            angles += angles[:1]
            
            ax = plt.subplot(2, 2, 4, projection='polar')
            ax.plot(angles, radar_values, 'o-', linewidth=2)
            ax.fill(angles, radar_values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_labels)
            ax.set_title('Performance Metrics Radar')
            ax.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = os.path.join(output_dir, f"evaluation_visualization_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到: {plot_path}")
    
    def _print_evaluation_summary(self, evaluation_results: Dict[str, Any]):
        """打印评估摘要"""
        metrics = evaluation_results.get('performance_metrics', {})
        strategy_analysis = evaluation_results.get('strategy_analysis', {})
        
        print("\n" + "="*80)
        print("评估摘要")
        print("="*80)
        
        # 性能指标
        print("\n📊 性能指标:")
        print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  索提诺比率: {metrics.get('sortino_ratio', 0):.3f}")
        print(f"  卡玛比率: {metrics.get('calmar_ratio', 0):.3f}")
        print(f"  最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"  年化波动率: {metrics.get('volatility', 0)*100:.2f}%")
        print(f"  胜率: {metrics.get('win_rate', 0)*100:.2f}%")
        
        # 收益指标
        returns = metrics.get('portfolio_returns', {})
        print(f"\n💰 收益指标:")
        print(f"  平均收益率: {returns.get('mean', 0)*100:.2f}%")
        print(f"  收益率标准差: {returns.get('std', 0)*100:.2f}%")
        
        # 策略分析
        print(f"\n🎯 策略分析:")
        strategy_performance = strategy_analysis.get('strategy_performance', {})
        for strategy, perf in strategy_performance.items():
            print(f"  {strategy}: {perf['percentage']:.1f}% (奖励: {perf['avg_reward']:.3f})")
        
        print(f"  策略切换次数: {strategy_analysis.get('total_strategy_switches', 0)}")
        print(f"  策略多样性: {strategy_analysis.get('strategy_diversity', 0)*100:.1f}%")
        
        print("="*80)
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """获取评估历史"""
        return self.evaluation_history
    
    def clear_evaluation_history(self):
        """清空评估历史"""
        self.evaluation_history = []
    
    def evaluate_long_term(self, episode_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        长期评估（文档要求的功能）
        
        文档要求评估4个指标：
        1. 日均切换次数
        2. 切换胜率
        3. 平均持仓时长
        4. 切换频率变异系数
        
        参数调整规则：
        1. 日均切换次数 > 10 → 增加切换惩罚
        2. 切换胜率 < 50% → 减少切换频率
        3. 平均持仓时长 < 5分钟 → 增加持仓奖励
        4. 切换频率变异系数 > 0.5 → 平滑切换行为
        """
        # 计算4个长期评估指标
        daily_switch_count = self._calculate_daily_switch_count(episode_data)
        switch_win_rate = self._calculate_switch_win_rate(episode_data)
        avg_holding_period = self._calculate_avg_holding_period(episode_data)
        switch_frequency_cv = self._calculate_switch_frequency_cv(episode_data)
        
        # 更新长期评估数据
        self.long_term_metrics['daily_switch_counts'].append(daily_switch_count)
        self.long_term_metrics['switch_win_rates'].append(switch_win_rate)
        self.long_term_metrics['avg_holding_periods'].append(avg_holding_period)
        self.long_term_metrics['switch_frequency_cv'].append(switch_frequency_cv)
        
        # 生成参数调整建议
        parameter_adjustments = self._generate_parameter_adjustments(
            daily_switch_count, switch_win_rate, avg_holding_period, switch_frequency_cv
        )
        
        # 记录参数调整
        self.parameter_adjustments.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'daily_switch_count': daily_switch_count,
                'switch_win_rate': switch_win_rate,
                'avg_holding_period': avg_holding_period,
                'switch_frequency_cv': switch_frequency_cv
            },
            'adjustments': parameter_adjustments
        })
        
        long_term_results = {
            'daily_switch_count': daily_switch_count,
            'switch_win_rate': switch_win_rate,
            'avg_holding_period': avg_holding_period,
            'switch_frequency_cv': switch_frequency_cv,
            'parameter_adjustments': parameter_adjustments,
            'long_term_trends': self._calculate_long_term_trends()
        }
        
        return long_term_results
    
    def _calculate_daily_switch_count(self, episode_data: List[Dict[str, Any]]) -> float:
        """计算日均切换次数"""
        total_switches = 0
        total_days = 0
        
        for ep in episode_data:
            switches_in_episode = 0
            prev_action = None
            
            for step in ep['history']:
                current_action = step['action']
                if prev_action is not None and current_action != prev_action:
                    switches_in_episode += 1
                prev_action = current_action
            
            # 假设每个episode代表一天
            total_switches += switches_in_episode
            total_days += 1
        
        return total_switches / total_days if total_days > 0 else 0
    
    def _calculate_switch_win_rate(self, episode_data: List[Dict[str, Any]]) -> float:
        """计算切换胜率（切换后N步收益为正的比例）"""
        successful_switches = 0
        total_switches = 0
        
        for ep in episode_data:
            for i in range(len(ep['history']) - 1):
                current_action = ep['history'][i]['action']
                next_action = ep['history'][i + 1]['action']
                
                # 检测切换
                if current_action != next_action:
                    total_switches += 1
                    
                    # 检查切换后N步的收益
                    lookahead_steps = min(5, len(ep['history']) - i - 1)
                    future_rewards = [ep['history'][i + j]['reward'] for j in range(1, lookahead_steps + 1)]
                    
                    # 如果未来平均收益为正，视为成功切换
                    if future_rewards and np.mean(future_rewards) > 0:
                        successful_switches += 1
        
        return successful_switches / total_switches if total_switches > 0 else 0
    
    def _calculate_avg_holding_period(self, episode_data: List[Dict[str, Any]]) -> float:
        """计算平均持仓时长（分钟）"""
        holding_periods = []
        
        for ep in episode_data:
            current_action = None
            action_start_time = 0
            
            for step_idx, step in enumerate(ep['history']):
                if current_action is None:
                    current_action = step['action']
                    action_start_time = step_idx
                elif step['action'] != current_action:
                    # 持仓结束
                    holding_period = step_idx - action_start_time
                    holding_periods.append(holding_period)
                    
                    # 开始新的持仓
                    current_action = step['action']
                    action_start_time = step_idx
            
            # 记录最后一个持仓
            if current_action is not None:
                holding_period = len(ep['history']) - action_start_time
                holding_periods.append(holding_period)
        
        return np.mean(holding_periods) if holding_periods else 0
    
    def _calculate_switch_frequency_cv(self, episode_data: List[Dict[str, Any]]) -> float:
        """计算切换频率变异系数"""
        switch_counts_per_episode = []
        
        for ep in episode_data:
            switches_in_episode = 0
            prev_action = None
            
            for step in ep['history']:
                current_action = step['action']
                if prev_action is not None and current_action != prev_action:
                    switches_in_episode += 1
                prev_action = current_action
            
            switch_counts_per_episode.append(switches_in_episode)
        
        if not switch_counts_per_episode or np.mean(switch_counts_per_episode) == 0:
            return 0
        
        return np.std(switch_counts_per_episode) / np.mean(switch_counts_per_episode)
    
    def _generate_parameter_adjustments(self, 
                                       daily_switch_count: float,
                                       switch_win_rate: float,
                                       avg_holding_period: float,
                                       switch_frequency_cv: float) -> Dict[str, Any]:
        """
        生成参数调整建议（文档要求的规则）
        
        规则：
        1. 日均切换次数 > 10 → 增加切换惩罚
        2. 切换胜率 < 50% → 减少切换频率
        3. 平均持仓时长 < 5分钟 → 增加持仓奖励
        4. 切换频率变异系数 > 0.5 → 平滑切换行为
        """
        adjustments = {}
        
        # 规则1：日均切换次数 > 10 → 增加切换惩罚
        if daily_switch_count > 10:
            adjustments['switch_penalty'] = {
                'current': 'low',
                'recommended': 'high',
                'reason': f'日均切换次数过高 ({daily_switch_count:.1f} > 10)',
                'adjustment': '增加切换惩罚系数'
            }
        
        # 规则2：切换胜率 < 50% → 减少切换频率
        if switch_win_rate < 0.5:
            adjustments['switch_frequency'] = {
                'current': 'normal',
                'recommended': 'low',
                'reason': f'切换胜率过低 ({switch_win_rate*100:.1f}% < 50%)',
                'adjustment': '减少切换频率阈值'
            }
        
        # 规则3：平均持仓时长 < 5分钟 → 增加持仓奖励
        if avg_holding_period < 5:
            adjustments['holding_reward'] = {
                'current': 'normal',
                'recommended': 'high',
                'reason': f'平均持仓时长过短 ({avg_holding_period:.1f}分钟 < 5分钟)',
                'adjustment': '增加持仓奖励系数'
            }
        
        # 规则4：切换频率变异系数 > 0.5 → 平滑切换行为
        if switch_frequency_cv > 0.5:
            adjustments['switch_smoothing'] = {
                'current': 'low',
                'recommended': 'high',
                'reason': f'切换频率变异系数过高 ({switch_frequency_cv:.2f} > 0.5)',
                'adjustment': '增加切换平滑参数'
            }
        
        return adjustments
    
    def _calculate_long_term_trends(self) -> Dict[str, Any]:
        """计算长期趋势"""
        trends = {}
        
        for metric_name, values in self.long_term_metrics.items():
            if isinstance(values, list) and len(values) >= 2:
                # 计算趋势（最近5个值的斜率）
                recent_values = values[-5:] if len(values) >= 5 else values
                x = np.arange(len(recent_values))
                slope, _ = np.polyfit(x, recent_values, 1)
                
                trends[metric_name] = {
                    'current': values[-1] if values else 0,
                    'trend': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
                    'slope': float(slope),
                    'history_length': len(values)
                }
        
        return trends
    
    def get_long_term_metrics(self) -> Dict[str, Any]:
        """获取长期评估指标"""
        return self.long_term_metrics
    
    def get_parameter_adjustments(self) -> List[Dict[str, Any]]:
        """获取参数调整记录"""
        return self.parameter_adjustments
    
    def clear_long_term_data(self):
        """清空长期评估数据"""
        self.long_term_metrics = {
            'daily_switch_counts': [],
            'switch_win_rates': [],
            'avg_holding_periods': [],
            'switch_frequency_cv': [],
            'market_state_win_rates': {},
            'switch_costs': [],
            'intraday_drawdown_freq': []
        }
        self.parameter_adjustments = []
    
    def __str__(self) -> str:
        return f"Evaluator(env={self.env}, agent={self.agent}, evaluations={len(self.evaluation_history)})"
    
    def __repr__(self) -> str:
        return self.__str__()
