"""
回测引擎 (Backtest Engine)

负责历史数据回测和性能分析
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import os
import random
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from src.environment.trading_env import TradingEnvironment
from src.agent.ddqn_agent import DDQNAgent
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor


class BacktestEngine:
    """
    回测引擎
    
    负责：
    1. 历史数据回测
    2. 回测结果分析
    3. 策略对比
    4. 参数敏感性分析
    """
    
    def __init__(self, 
                 data_path: str,
                 symbols: List[str],
                 initial_capital: float = 1000000,
                 period_minutes: int = 20,
                 bar_frequency: str = 'minute'):
        """
        初始化回测引擎
        
        Args:
            data_path: 数据路径
            symbols: 股票代码列表
            initial_capital: 初始资金
            period_minutes: 基础周期（分钟）
            bar_frequency: K线频率（minute 或 daily）
        """
        self.data_path = data_path
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.period_minutes = period_minutes
        self.bar_frequency = str(bar_frequency).lower()
        
        # 数据加载器和预处理器
        self.data_loader = DataLoader(data_path, symbols)
        self.data_preprocessor = DataPreprocessor(
            period_minutes,
            bar_frequency=self.bar_frequency
        )
        
        # 回测结果存储
        self.backtest_results = {}
        self.comparison_results = {}
        
        # 输出目录
        self.output_dir = "outputs/backtest"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_and_prepare_data(self, 
                             start_date: str,
                             end_date: str,
                             train_ratio: float = 0.7) -> Dict[str, pd.DataFrame]:
        """
        加载和准备回测数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            train_ratio: 训练集比例
        
        Returns:
            data_dict: 包含训练、验证、测试数据的字典
        """
        print(f"加载数据: {start_date} 到 {end_date}")
        
        # 加载原始数据
        raw_data = self.data_loader.load_data(start_date, end_date)
        
        if raw_data.empty:
            raise ValueError(f"在 {self.data_path} 中未找到 {start_date} 到 {end_date} 的数据")
        
        print(f"原始数据形状: {raw_data.shape}")
        print(f"数据时间范围: {raw_data['timestamp'].min()} 到 {raw_data['timestamp'].max()}")
        
        # 数据预处理
        print("数据预处理...")
        processed_data = self.data_preprocessor.preprocess_pipeline(
            raw_data,
            handle_missing=True,
            aggregate=True,
            remove_outliers=True,
            add_time_features=False,
            calculate_returns=True
        )
        
        print(f"处理后数据形状: {processed_data.shape}")
        
        # 划分数据集
        print("划分数据集...")
        train_data, val_data, test_data = self.data_loader.split_data(
            processed_data, 
            train_ratio=train_ratio,
            val_ratio=0.15
        )
        
        print(f"训练集: {train_data.shape}, 验证集: {val_data.shape}, 测试集: {test_data.shape}")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'full': processed_data
        }

    def _create_backtest_env(self, test_data: pd.DataFrame, env_config: Dict[str, Any]) -> TradingEnvironment:
        """按统一配置创建回测环境"""
        return TradingEnvironment(
            data=test_data,
            initial_capital=env_config['initial_capital'],
            period_minutes=env_config['period_minutes'],
            state_lookback_days=env_config.get('state_lookback_days', 5),
            switch_period_multiplier=env_config.get('switch_period_multiplier', 1),
            transaction_cost=env_config.get('transaction_cost', 0.001),
            dynamic_switch=env_config.get('dynamic_switch', True),
            state_update_frequency=env_config.get('state_update_frequency', 'dynamic'),
            allow_short=env_config.get('allow_short', True),
            max_short_per_symbol_ratio=env_config.get('max_short_per_symbol_ratio', 0.1),
            max_total_short_ratio=env_config.get('max_total_short_ratio', 0.25),
            max_position_ratio=env_config.get('max_position_ratio', 0.3),
            total_position_ratio=env_config.get('total_position_ratio', 0.8),
            rebalance_tolerance=env_config.get('rebalance_tolerance', 0.02),
            max_trade_fraction=env_config.get('max_trade_fraction', 0.01),
            slippage_rate=env_config.get('slippage_rate', 0.0002),
            impact_rate=env_config.get('impact_rate', 0.0001),
            min_commission=env_config.get('min_commission', 2.0),
            min_signal_threshold=env_config.get('min_signal_threshold', 0.08),
            signal_power=env_config.get('signal_power', 1.5),
            profit_lock_enabled=env_config.get('profit_lock_enabled', False),
            profit_lock_min_return=env_config.get('profit_lock_min_return', 0.03),
            profit_lock_drawdown_threshold=env_config.get('profit_lock_drawdown_threshold', 0.08),
            profit_lock_cooldown_steps=env_config.get('profit_lock_cooldown_steps', 10),
            profit_lock_safe_action=env_config.get('profit_lock_safe_action', 0),
            min_action_hold_steps=env_config.get('min_action_hold_steps', 1)
        )

    @staticmethod
    def _extract_step_market_timestamp(env: TradingEnvironment, step_idx: Any) -> str:
        """从环境中提取当前step对应的市场时间"""
        try:
            step_idx = int(step_idx)
        except Exception:
            step_idx = 0

        # 优先使用环境按“唯一时间点”构建的时间轴（与step语义一致）
        # 避免把step当作period_data行号，导致多股票场景下时间轴被压缩到少量日期。
        unique_ts = getattr(env, 'unique_timestamps', None)
        if unique_ts is not None and len(unique_ts) > 0:
            clamped_idx = min(max(step_idx, 0), len(unique_ts) - 1)
            ts = pd.to_datetime(unique_ts[clamped_idx], errors='coerce')
            return ts.isoformat() if pd.notna(ts) else ''

        # 回退：兼容旧环境
        if hasattr(env, 'period_data') and 'timestamp' in env.period_data.columns and len(env.period_data) > 0:
            clamped_idx = min(max(step_idx, 0), len(env.period_data) - 1)
            ts_val = env.period_data.iloc[clamped_idx]['timestamp']
            ts = pd.to_datetime(ts_val, errors='coerce')
            return ts.isoformat() if pd.notna(ts) else ''
        return ''

    def _simulate_policy_curve(self,
                               test_data: pd.DataFrame,
                               env_config: Dict[str, Any],
                               policy_name: str) -> Dict[str, Any]:
        """
        在同一测试集上运行基准策略并返回累计收益曲线

        policy_name 取值：
        - fixed_{id}
        - random
        - rotation
        - majority_voting
        - strategy_ensemble
        """
        env = self._create_backtest_env(test_data, env_config)
        state = env.reset()
        done = False
        step_count = 0
        portfolio_values = []
        actions_history = []
        market_timestamps = []

        rng = random.Random(42)
        vote_history: List[int] = []
        ensemble_scores = np.zeros(6, dtype=float)

        while not done:
            if policy_name.startswith('fixed_'):
                action = int(policy_name.split('_')[-1])
            elif policy_name == 'random':
                action = rng.randint(0, 5)
            elif policy_name == 'rotation':
                action = step_count % 6
            elif policy_name == 'majority_voting':
                raw_state = env._get_raw_state()
                signals = raw_state.get('strategy_signals', {}) if isinstance(raw_state, dict) else {}
                top_action = int(max(range(6), key=lambda i: float(signals.get(i, 0.0))))
                vote_history.append(top_action)
                window_votes = vote_history[-15:]
                counter = Counter(window_votes)
                max_votes = max(counter.values())
                candidates = [k for k, v in counter.items() if v == max_votes]
                action = int(min(candidates))
            elif policy_name == 'strategy_ensemble':
                raw_state = env._get_raw_state()
                signals = raw_state.get('strategy_signals', {}) if isinstance(raw_state, dict) else {}
                signal_vec = np.array([float(signals.get(i, 0.0)) for i in range(6)], dtype=float)
                ensemble_scores = 0.8 * ensemble_scores + 0.2 * signal_vec
                action = int(np.argmax(ensemble_scores))
            else:
                action = 0

            next_state, reward, done, info = env.step(action)
            state = next_state
            step_count += 1

            portfolio_values.append(info.get('portfolio_value', 0.0))
            actions_history.append(action)
            market_timestamps.append(self._extract_step_market_timestamp(env, info.get('step', step_count)))

        initial_capital = env_config.get('initial_capital', self.initial_capital)
        cumulative_returns = [
            (pv - initial_capital) / initial_capital if initial_capital > 0 else 0.0
            for pv in portfolio_values
        ]

        return {
            'policy_name': policy_name,
            'portfolio_values': portfolio_values,
            'actions_history': actions_history,
            'market_timestamps': market_timestamps,
            'cumulative_returns': cumulative_returns,
            'final_return': cumulative_returns[-1] if cumulative_returns else 0.0
        }

    def _compute_buy_and_hold_curve(self,
                                    test_data: pd.DataFrame,
                                    target_market_timestamps: List[str]) -> Dict[str, Any]:
        """计算等权买入并持有（Buy & Hold）累计收益曲线"""
        if test_data.empty:
            return {
                'cumulative_returns': [],
                'market_timestamps': target_market_timestamps,
                'final_return': 0.0
            }

        price_table = test_data.pivot_table(
            index='timestamp', columns='symbol', values='close', aggfunc='last'
        ).sort_index().ffill()
        # 防御性去重：避免后续reindex因重复标签报错
        price_table = price_table[~price_table.index.duplicated(keep='last')]
        price_table = price_table.loc[:, ~price_table.columns.duplicated(keep='last')]

        if price_table.empty:
            return {
                'cumulative_returns': [],
                'market_timestamps': target_market_timestamps,
                'final_return': 0.0
            }

        ts_series = pd.to_datetime(pd.Series(target_market_timestamps), errors='coerce')
        valid_ts = pd.DatetimeIndex(ts_series.dropna())
        if len(valid_ts) == 0:
            aligned_prices = price_table
            output_index = aligned_prices.index
        else:
            unique_ts = pd.DatetimeIndex(pd.unique(valid_ts))
            aligned_unique = (
                price_table.reindex(price_table.index.union(valid_ts))
                .sort_index()
                .ffill()
            )
            aligned_unique = aligned_unique[~aligned_unique.index.duplicated(keep='last')]
            aligned_unique = aligned_unique.reindex(unique_ts)
            if aligned_unique.empty:
                return {
                    'cumulative_returns': [],
                    'market_timestamps': target_market_timestamps,
                    'final_return': 0.0
                }

            base_prices = aligned_unique.iloc[0].replace(0, np.nan)
            relative = aligned_unique.divide(base_prices, axis=1)
            cumulative_unique = (relative.mean(axis=1, skipna=True) - 1.0).ffill().fillna(0.0)
            cumulative_map = cumulative_unique.to_dict()

            # 保留原始时间戳长度（含重复），便于与主策略曲线一一对齐
            cumulative_list = []
            last_val = 0.0
            for ts in ts_series:
                if pd.isna(ts):
                    cumulative_list.append(last_val)
                    continue
                val = float(cumulative_map.get(ts, last_val))
                cumulative_list.append(val)
                last_val = val

            return {
                'cumulative_returns': cumulative_list,
                'market_timestamps': target_market_timestamps,
                'final_return': float(cumulative_list[-1]) if cumulative_list else 0.0
            }

        if aligned_prices.empty:
            return {
                'cumulative_returns': [],
                'market_timestamps': target_market_timestamps,
                'final_return': 0.0
            }

        base_prices = aligned_prices.iloc[0].replace(0, np.nan)
        relative = aligned_prices.divide(base_prices, axis=1)
        cumulative = relative.mean(axis=1, skipna=True) - 1.0
        cumulative = cumulative.ffill().fillna(0.0)

        return {
            'cumulative_returns': cumulative.tolist(),
            'market_timestamps': [t.isoformat() for t in output_index],
            'final_return': float(cumulative.iloc[-1]) if len(cumulative) else 0.0
        }

    def _compute_benchmark_curves(self,
                                  test_data: pd.DataFrame,
                                  env_config: Dict[str, Any],
                                  base_market_timestamps: List[str]) -> Dict[str, Dict[str, Any]]:
        """计算测试集多基准策略曲线，用于与RL主策略对比"""
        benchmarks: Dict[str, Dict[str, Any]] = {}

        # Buy & Hold（等权）
        benchmarks['Buy & Hold'] = self._compute_buy_and_hold_curve(test_data, base_market_timestamps)

        # Best Strategy（从6个固定策略中选测试集表现最佳）
        fixed_results = []
        for i in range(6):
            result = self._simulate_policy_curve(test_data, env_config, f'fixed_{i}')
            result['strategy_id'] = i
            fixed_results.append(result)
        if fixed_results:
            best_result = max(fixed_results, key=lambda x: x.get('final_return', -np.inf))
            benchmarks['Best Strategy'] = {
                'cumulative_returns': best_result.get('cumulative_returns', []),
                'market_timestamps': best_result.get('market_timestamps', []),
                'final_return': best_result.get('final_return', 0.0),
                'strategy_id': best_result.get('strategy_id', 0)
            }

        # 其他基准策略
        for label, policy_name in [
            ('Random Strategy', 'random'),
            ('Fixed Rotation', 'rotation'),
            ('Majority Voting', 'majority_voting'),
            ('Strategy Ensemble', 'strategy_ensemble')
        ]:
            result = self._simulate_policy_curve(test_data, env_config, policy_name)
            benchmarks[label] = {
                'cumulative_returns': result.get('cumulative_returns', []),
                'market_timestamps': result.get('market_timestamps', []),
                'final_return': result.get('final_return', 0.0)
            }

        return benchmarks
    
    def run_backtest(self,
                    agent: DDQNAgent,
                    test_data: pd.DataFrame,
                    env_config: Dict[str, Any] = None,
                    save_results: bool = True,
                    result_name: str = None) -> Dict[str, Any]:
        """
        运行单次回测
        
        Args:
            agent: 训练好的DDQN智能体
            test_data: 测试数据
            env_config: 环境配置
            save_results: 是否保存结果
            result_name: 结果名称
        
        Returns:
            backtest_result: 回测结果
        """
        if env_config is None:
            env_config = {
                'initial_capital': self.initial_capital,
                'period_minutes': self.period_minutes,
                'bar_frequency': self.bar_frequency,
                'state_lookback_days': 5,
                'switch_period_multiplier': 1,
                'transaction_cost': 0.001,
                'min_action_hold_steps': 1
            }
        
        print(f"开始回测: {result_name or 'unnamed'}")
        print(f"测试数据形状: {test_data.shape}")
        print(f"环境配置: {env_config}")
        
        # 创建交易环境
        env = self._create_backtest_env(test_data, env_config)
        
        # 运行回测
        state = env.reset()
        done = False
        step_count = 0
        portfolio_values = []
        actions_history = []
        raw_actions_history = []
        strategy_names_history = []
        rewards_history = []
        timestamps = []
        market_timestamps = []
        steps_advanced_history = []
        
        while not done:
            # 选择动作（评估模式）
            action = agent.select_action(state, training=False)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 记录数据
            raw_action = int(info.get('raw_action', action))
            effective_action = int(info.get('effective_action', raw_action))
            portfolio_values.append(info.get('portfolio_value', 0))
            raw_actions_history.append(raw_action)
            actions_history.append(effective_action)
            strategy_names_history.append(info.get('strategy', f'Strategy_{effective_action}'))
            rewards_history.append(reward)
            timestamps.append(step_count)
            steps_advanced_history.append(int(info.get('steps_advanced', 1)))

            # 记录真实市场时间（用于年化指标计算和时间轴绘图）
            market_timestamps.append(self._extract_step_market_timestamp(env, info.get('step', self.period_minutes)))
            
            # 更新状态
            state = next_state
            step_count += 1
            
            # 进度显示
            if step_count % 100 == 0:
                print(f"  步骤 {step_count}, 组合价值: {portfolio_values[-1]:,.2f}")
        
        # 计算回测指标
        backtest_metrics = self._calculate_backtest_metrics(
            portfolio_values, 
            actions_history, 
            rewards_history,
            env_config['initial_capital'],
            market_timestamps=market_timestamps,
            steps_advanced_history=steps_advanced_history,
            period_minutes=env_config.get('period_minutes', self.period_minutes),
            bar_frequency=env_config.get('bar_frequency', self.bar_frequency)
        )
        
        cumulative_returns = []
        if portfolio_values:
            cumulative_returns = [
                (pv - env_config['initial_capital']) / env_config['initial_capital']
                for pv in portfolio_values
            ]

        # 计算基准策略曲线（用于累计收益对比）
        benchmark_curves = self._compute_benchmark_curves(
            test_data=test_data,
            env_config=env_config,
            base_market_timestamps=market_timestamps
        )

        # 收集详细信息
        backtest_result = {
            'result_name': result_name or f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'portfolio_values': portfolio_values,
            'actions_history': actions_history,
            'raw_actions_history': raw_actions_history,
            'strategy_names_history': strategy_names_history,
            'rewards_history': rewards_history,
            'cumulative_returns': cumulative_returns,
            'benchmark_curves': benchmark_curves,
            'timestamps': timestamps,
            'market_timestamps': market_timestamps,
            'steps_advanced_history': steps_advanced_history,
            'initial_capital': env_config['initial_capital'],
            'final_portfolio_value': portfolio_values[-1] if portfolio_values else 0,
            'total_return': (portfolio_values[-1] - env_config['initial_capital']) / env_config['initial_capital'] if portfolio_values else 0,
            'total_steps': step_count,
            'metrics': backtest_metrics,
            'env_config': env_config,
            'backtest_timestamp': datetime.now().isoformat()
        }
        
        # 保存结果
        if save_results:
            self._save_backtest_result(backtest_result)
        
        # 存储结果
        result_key = result_name or f"result_{len(self.backtest_results)}"
        self.backtest_results[result_key] = backtest_result
        
        print(f"回测完成: {result_key}")
        print(f"  初始资金: {env_config['initial_capital']:,.2f}")
        print(f"  最终资金: {portfolio_values[-1]:,.2f}")
        print(f"  总收益率: {backtest_result['total_return']*100:.2f}%")
        print(f"  夏普比率: {backtest_metrics.get('sharpe_ratio', 0):.3f}")
        
        return backtest_result
    
    def run_multiple_backtests(self,
                              agent: DDQNAgent,
                              test_data_dict: Dict[str, pd.DataFrame],
                              env_config: Dict[str, Any] = None,
                              save_results: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        运行多个回测（不同时间段）
        
        Args:
            agent: 训练好的DDQN智能体
            test_data_dict: 不同时间段的测试数据字典
            env_config: 环境配置
            save_results: 是否保存结果
        
        Returns:
            all_results: 所有回测结果
        """
        all_results = {}
        
        for period_name, test_data in test_data_dict.items():
            print(f"\n运行回测: {period_name}")
            
            try:
                result = self.run_backtest(
                    agent=agent,
                    test_data=test_data,
                    env_config=env_config,
                    save_results=save_results,
                    result_name=period_name
                )
                all_results[period_name] = result
            except Exception as e:
                print(f"  回测 {period_name} 失败: {e}")
                continue
        
        # 比较多个回测结果
        if len(all_results) > 1:
            comparison = self._compare_backtest_results(all_results)
            self.comparison_results = comparison
            
            if save_results:
                self._save_comparison_results(comparison)
        
        return all_results
    
    def run_strategy_comparison(self,
                               test_data: pd.DataFrame,
                               strategies: Dict[str, Any],
                               env_config: Dict[str, Any] = None,
                               save_results: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        运行策略对比回测
        
        Args:
            test_data: 测试数据
            strategies: 策略字典 {策略名: 策略对象或配置}
            env_config: 环境配置
            save_results: 是否保存结果
        
        Returns:
            comparison_results: 策略对比结果
        """
        print(f"开始策略对比回测，共 {len(strategies)} 种策略")
        
        strategy_results = {}
        
        for strategy_name, strategy_config in strategies.items():
            print(f"\n测试策略: {strategy_name}")
            
            # 这里可以根据策略配置创建不同的智能体或环境
            # 目前使用默认的DDQN智能体，但可以扩展为不同的策略
            
            # 创建智能体（这里简化处理，实际应根据策略配置创建）
            from src.agent.ddqn_agent import DDQNAgent
            
            # 获取状态维度（需要从环境中获取）
            env = TradingEnvironment(
                data=test_data,
                initial_capital=env_config.get('initial_capital', self.initial_capital) if env_config else self.initial_capital,
                period_minutes=env_config.get('period_minutes', self.period_minutes) if env_config else self.period_minutes
            )
            
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            
            # 创建智能体（这里可以加载预训练模型或使用随机策略）
            agent = DDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=0.0001,
                epsilon_start=0.1,  # 评估模式下使用较低的探索率
                epsilon_end=0.01,
                device='cpu'
            )
            
            # 运行回测
            try:
                result = self.run_backtest(
                    agent=agent,
                    test_data=test_data,
                    env_config=env_config,
                    save_results=False,  # 先不单独保存
                    result_name=strategy_name
                )
                strategy_results[strategy_name] = result
            except Exception as e:
                print(f"  策略 {strategy_name} 回测失败: {e}")
                continue
        
        # 策略对比分析
        comparison = self._compare_strategies(strategy_results)
        
        if save_results:
            self._save_strategy_comparison(comparison, strategy_results)
        
        return comparison
    
    def run_parameter_sensitivity_analysis(self,
                                         agent: DDQNAgent,
                                         test_data: pd.DataFrame,
                                         parameter_grid: Dict[str, List[Any]],
                                         env_config: Dict[str, Any] = None,
                                         save_results: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        运行参数敏感性分析
        
        Args:
            agent: 基础智能体
            test_data: 测试数据
            parameter_grid: 参数网格 {参数名: [参数值列表]}
            env_config: 环境配置
            save_results: 是否保存结果
        
        Returns:
            sensitivity_results: 敏感性分析结果
        """
        print(f"开始参数敏感性分析")
        print(f"参数网格: {parameter_grid}")
        
        sensitivity_results = {}
        
        # 生成所有参数组合
        from itertools import product
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(product(*param_values))
        
        print(f"共 {len(param_combinations)} 种参数组合")
        
        for i, param_combo in enumerate(param_combinations):
            param_dict = dict(zip(param_names, param_combo))
            combo_name = "_".join([f"{k}_{v}" for k, v in param_dict.items()])
            
            print(f"\n测试参数组合 {i+1}/{len(param_combinations)}: {combo_name}")
            
            # 更新环境配置
            current_env_config = env_config.copy() if env_config else {}
            current_env_config.update(param_dict)
            
            # 运行回测
            try:
                result = self.run_backtest(
                    agent=agent,
                    test_data=test_data,
                    env_config=current_env_config,
                    save_results=False,
                    result_name=combo_name
                )
                sensitivity_results[combo_name] = {
                    'parameters': param_dict,
                    'result': result
                }
            except Exception as e:
                print(f"  参数组合 {combo_name} 回测失败: {e}")
                continue
        
        # 敏感性分析
        analysis = self._analyze_parameter_sensitivity(sensitivity_results)
        
        if save_results:
            self._save_sensitivity_analysis(analysis, sensitivity_results)
        
        return analysis
    
    def _calculate_backtest_metrics(self,
                                  portfolio_values: List[float],
                                  actions_history: List[int],
                                  rewards_history: List[float],
                                  initial_capital: float,
                                  market_timestamps: List[str] = None,
                                  steps_advanced_history: List[int] = None,
                                  period_minutes: int = 20,
                                  bar_frequency: str = 'minute') -> Dict[str, Any]:
        """
        计算回测指标
        
        Args:
            portfolio_values: 组合价值序列
            actions_history: 动作历史
            rewards_history: 奖励历史
            initial_capital: 初始资金
        
        Returns:
            metrics: 回测指标
        """
        if not portfolio_values:
            return {}
        
        # 计算收益率序列
        returns = []
        for i in range(1, len(portfolio_values)):
            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            returns.append(ret)
        
        # 基本指标
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital

        period_minutes = max(int(period_minutes), 1)
        bar_frequency = str(bar_frequency).lower()

        # 年化基准：日频固定252；分钟频按A股240分钟/日换算
        if bar_frequency == 'daily':
            base_periods_per_year = 252.0
        else:
            base_periods_per_year = 252.0 * (240.0 / period_minutes)

        # 年化基准：优先使用真实时间跨度；否则回退到按交易周期推算
        periods_per_year = 252.0
        annualization_basis = 'fallback_252'
        years_span = 0.0

        # 优先按“实际跨越交易周期”年化（适合动态步长环境）
        if steps_advanced_history:
            total_advanced_steps = float(sum(max(int(s), 0) for s in steps_advanced_history))
            if total_advanced_steps > 0:
                # 每个advanced step对应一个period_minutes周期
                years_span = total_advanced_steps / base_periods_per_year
                periods_per_year = len(portfolio_values) / years_span
                annualization_basis = 'trading_steps_period'

        # 其次按真实时间戳年化
        if annualization_basis == 'fallback_252' and market_timestamps and len(market_timestamps) >= 2:
            ts = pd.to_datetime(market_timestamps, errors='coerce')
            ts = pd.Series(ts).dropna()
            if len(ts) >= 2:
                delta_days = (ts.iloc[-1] - ts.iloc[0]).days
                if delta_days > 0:
                    years_span = delta_days / 365.25
                    periods_per_year = len(portfolio_values) / years_span
                    annualization_basis = 'timestamp_span'

        if years_span > 0 and initial_capital > 0 and portfolio_values[-1] > 0:
            # CAGR
            annualized_return = (portfolio_values[-1] / initial_capital) ** (1.0 / years_span) - 1.0
        else:
            # 回退：线性近似（兼容旧逻辑）
            annualized_return = total_return * periods_per_year / max(len(portfolio_values), 1)
        
        # 风险指标
        volatility = np.std(returns) * np.sqrt(periods_per_year) if returns else 0
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # 风险调整指标
        sharpe_ratio = 0
        if returns and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
        
        sortino_ratio = 0
        if returns:
            downside_returns = [r for r in returns if r < 0]
            if downside_returns and np.std(downside_returns) > 0:
                sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(periods_per_year)

        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # 交易行为指标
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        avg_trade_return = np.mean(returns) if returns else 0
        profit_factor = abs(sum(r for r in returns if r > 0) / sum(r for r in returns if r < 0)) if sum(r for r in returns if r < 0) != 0 else 0
        
        # 策略切换指标
        strategy_switches = sum(1 for i in range(1, len(actions_history)) if actions_history[i] != actions_history[i-1])
        strategy_switch_rate = strategy_switches / len(actions_history) if actions_history else 0
        
        metrics = {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'periods_per_year': float(periods_per_year),
            'annualization_basis': annualization_basis,
            'bar_frequency': bar_frequency,
            'timespan_years': float(years_span),
            'volatility': float(volatility),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'win_rate': float(win_rate),
            'avg_trade_return': float(avg_trade_return),
            'profit_factor': float(profit_factor),
            'strategy_switches': strategy_switches,
            'strategy_switch_rate': float(strategy_switch_rate),
            'total_trades': len(returns),
            'final_portfolio_value': float(portfolio_values[-1]) if portfolio_values else 0
        }
        
        return metrics
    
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
    
    def _compare_backtest_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """比较多个回测结果"""
        comparison = {
            'summary': {},
            'rankings': {},
            'statistics': {}
        }
        
        # 收集所有指标
        all_metrics = {}
        for result_name, result in results.items():
            metrics = result.get('metrics', {})
            all_metrics[result_name] = metrics
        
        # 计算排名
        for metric_name in ['total_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate']:
            metric_values = {}
            for result_name, metrics in all_metrics.items():
                if metric_name in metrics:
                    metric_values[result_name] = metrics[metric_name]
            
            if metric_values:
                # 按指标值排序
                sorted_items = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                comparison['rankings'][metric_name] = dict(sorted_items)
        
        # 计算统计信息
        for metric_name in ['total_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
            values = []
            for metrics in all_metrics.values():
                if metric_name in metrics:
                    values.append(metrics[metric_name])
            
            if values:
                comparison['statistics'][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        # 生成摘要
        comparison['summary'] = {
            'total_results': len(results),
            'best_performer': max(results.items(), key=lambda x: x[1].get('metrics', {}).get('total_return', 0))[0] if results else 'N/A',
            'worst_performer': min(results.items(), key=lambda x: x[1].get('metrics', {}).get('total_return', 0))[0] if results else 'N/A',
            'most_consistent': self._find_most_consistent_result(results)
        }
        
        return comparison
    
    def _compare_strategies(self, strategy_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """比较不同策略"""
        comparison = {
            'performance_comparison': {},
            'strategy_rankings': {},
            'recommendations': []
        }
        
        # 性能比较
        for strategy_name, result in strategy_results.items():
            metrics = result.get('metrics', {})
            comparison['performance_comparison'][strategy_name] = {
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'final_portfolio_value': result.get('final_portfolio_value', 0)
            }
        
        # 策略排名
        for metric_name in ['total_return', 'sharpe_ratio']:
            metric_values = {}
            for strategy_name, result in strategy_results.items():
                metrics = result.get('metrics', {})
                if metric_name in metrics:
                    metric_values[strategy_name] = metrics[metric_name]
            
            if metric_values:
                sorted_items = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                comparison['strategy_rankings'][metric_name] = dict(sorted_items)
        
        # 生成推荐
        if strategy_results:
            # 按夏普比率推荐
            sharpe_ranking = comparison['strategy_rankings'].get('sharpe_ratio', {})
            if sharpe_ranking:
                best_sharpe = list(sharpe_ranking.keys())[0]
                comparison['recommendations'].append(
                    f"最佳风险调整策略: {best_sharpe} (夏普比率: {sharpe_ranking[best_sharpe]:.3f})"
                )
            
            # 按总收益推荐
            return_ranking = comparison['strategy_rankings'].get('total_return', {})
            if return_ranking:
                best_return = list(return_ranking.keys())[0]
                comparison['recommendations'].append(
                    f"最佳收益策略: {best_return} (总收益: {return_ranking[best_return]*100:.2f}%)"
                )
            
            # 按回撤控制推荐
            drawdown_values = {}
            for strategy_name, result in strategy_results.items():
                metrics = result.get('metrics', {})
                drawdown_values[strategy_name] = metrics.get('max_drawdown', 1.0)
            
            if drawdown_values:
                best_drawdown = min(drawdown_values.items(), key=lambda x: x[1])[0]
                comparison['recommendations'].append(
                    f"最佳风险控制策略: {best_drawdown} (最大回撤: {drawdown_values[best_drawdown]*100:.2f}%)"
                )
        
        return comparison
    
    def _analyze_parameter_sensitivity(self, sensitivity_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """分析参数敏感性"""
        analysis = {
            'parameter_importance': {},
            'optimal_parameters': {},
            'sensitivity_analysis': {}
        }
        
        # 提取参数和性能数据
        param_data = []
        for combo_name, data in sensitivity_results.items():
            params = data['parameters']
            result = data['result']
            metrics = result.get('metrics', {})
            
            param_data.append({
                'parameters': params,
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0)
            })
        
        if not param_data:
            return analysis
        
        # 分析每个参数的重要性
        param_names = list(param_data[0]['parameters'].keys())
        
        for param_name in param_names:
            # 收集该参数不同值对应的性能
            param_values = {}
            for data in param_data:
                param_value = str(data['parameters'][param_name])
                if param_value not in param_values:
                    param_values[param_value] = []
                param_values[param_value].append(data['total_return'])
            
            # 计算参数值对性能的影响
            if param_values:
                avg_returns = {}
                for value, returns in param_values.items():
                    avg_returns[value] = np.mean(returns) if returns else 0
                
                # 计算参数重要性（性能变化范围）
                if avg_returns:
                    max_return = max(avg_returns.values())
                    min_return = min(avg_returns.values())
                    importance = max_return - min_return
                    
                    analysis['parameter_importance'][param_name] = {
                        'importance': float(importance),
                        'optimal_value': max(avg_returns.items(), key=lambda x: x[1])[0],
                        'value_performance': avg_returns
                    }
        
        # 找到最优参数组合
        best_combo = max(sensitivity_results.items(), 
                        key=lambda x: x[1]['result'].get('metrics', {}).get('total_return', 0))
        
        analysis['optimal_parameters'] = {
            'combination': best_combo[0],
            'parameters': best_combo[1]['parameters'],
            'performance': best_combo[1]['result'].get('metrics', {})
        }
        
        # 敏感性分析
        for param_name in param_names:
            sensitivity_values = {}
            for data in param_data:
                param_value = str(data['parameters'][param_name])
                if param_value not in sensitivity_values:
                    sensitivity_values[param_value] = []
                sensitivity_values[param_value].append(data['total_return'])
            
            if sensitivity_values:
                analysis['sensitivity_analysis'][param_name] = {
                    'mean_performance': {k: float(np.mean(v)) for k, v in sensitivity_values.items()},
                    'std_performance': {k: float(np.std(v)) for k, v in sensitivity_values.items()},
                    'performance_range': {
                        'min': min([np.mean(v) for v in sensitivity_values.values()]),
                        'max': max([np.mean(v) for v in sensitivity_values.values()])
                    }
                }
        
        return analysis
    
    def _find_most_consistent_result(self, results: Dict[str, Dict[str, Any]]) -> str:
        """找到最稳定的回测结果"""
        if not results:
            return "N/A"
        
        consistency_scores = {}
        
        for result_name, result in results.items():
            metrics = result.get('metrics', {})
            
            # 计算一致性分数（基于多个指标的综合表现）
            score = 0
            if 'sharpe_ratio' in metrics:
                score += metrics['sharpe_ratio'] * 0.3
            if 'sortino_ratio' in metrics:
                score += metrics['sortino_ratio'] * 0.3
            if 'calmar_ratio' in metrics:
                score += metrics['calmar_ratio'] * 0.2
            if 'win_rate' in metrics:
                score += metrics['win_rate'] * 0.2
            
            consistency_scores[result_name] = score
        
        if consistency_scores:
            return max(consistency_scores.items(), key=lambda x: x[1])[0]
        
        return "N/A"
    
    def _save_backtest_result(self, backtest_result: Dict[str, Any]):
        """保存回测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_name = backtest_result.get('result_name', f"backtest_{timestamp}")
        
        # 保存为JSON
        json_path = os.path.join(self.output_dir, f"{result_name}.json")
        
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
        
        serializable_result = convert_to_serializable(backtest_result)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        print(f"回测结果已保存到: {json_path}")
        
        # 生成可视化图表
        self._generate_backtest_visualization(backtest_result, result_name)
    
    def _save_comparison_results(self, comparison: Dict[str, Any]):
        """保存比较结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.output_dir, f"comparison_{timestamp}.json")
        
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
        
        serializable_comparison = convert_to_serializable(comparison)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_comparison, f, indent=2, ensure_ascii=False)
        
        print(f"比较结果已保存到: {json_path}")
        
        # 生成比较可视化
        self._generate_comparison_visualization(comparison, timestamp)
    
    def _save_strategy_comparison(self, comparison: Dict[str, Any], strategy_results: Dict[str, Dict[str, Any]]):
        """保存策略对比结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存对比结果
        comparison_path = os.path.join(self.output_dir, f"strategy_comparison_{timestamp}.json")
        
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
        
        serializable_data = {
            'comparison': convert_to_serializable(comparison),
            'strategy_results': convert_to_serializable(strategy_results)
        }
        
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"策略对比结果已保存到: {comparison_path}")
        
        # 生成策略对比可视化
        self._generate_strategy_comparison_visualization(comparison, strategy_results, timestamp)
    
    def _save_sensitivity_analysis(self, analysis: Dict[str, Any], sensitivity_results: Dict[str, Dict[str, Any]]):
        """保存敏感性分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存分析结果
        analysis_path = os.path.join(self.output_dir, f"sensitivity_analysis_{timestamp}.json")
        
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
        
        serializable_data = {
            'analysis': convert_to_serializable(analysis),
            'sensitivity_results': convert_to_serializable(sensitivity_results)
        }
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"敏感性分析结果已保存到: {analysis_path}")
        
        # 生成敏感性分析可视化
        self._generate_sensitivity_visualization(analysis, timestamp)

    @staticmethod
    def _resolve_plot_x_axis(market_timestamps: List[str], expected_len: int):
        """将时间戳序列解析为可绘图x轴，解析失败时回退为步数"""
        if market_timestamps and len(market_timestamps) == expected_len:
            ts = pd.to_datetime(pd.Series(market_timestamps), errors='coerce')
            if ts.notna().sum() >= max(3, int(0.8 * expected_len)):
                return pd.DatetimeIndex(ts).to_pydatetime().tolist(), True
        return list(range(expected_len)), False

    def _plot_cumulative_return_with_benchmarks(self, ax, backtest_result: Dict[str, Any]):
        """绘制主策略+基准策略累计收益率曲线"""
        initial_capital = backtest_result.get('initial_capital', 0)
        portfolio_values = backtest_result.get('portfolio_values', [])
        market_timestamps = backtest_result.get('market_timestamps', [])
        benchmark_curves = backtest_result.get('benchmark_curves', {})

        if not portfolio_values or initial_capital <= 0:
            return False

        main_curve = [
            (pv - initial_capital) / initial_capital * 100 for pv in portfolio_values
        ]
        x_main, is_datetime = self._resolve_plot_x_axis(market_timestamps, len(main_curve))

        ax.plot(x_main, main_curve, color='tab:blue', linewidth=2.0, label='RL Agent')

        benchmark_order = [
            'Buy & Hold',
            'Best Strategy',
            'Random Strategy',
            'Fixed Rotation',
            'Majority Voting',
            'Strategy Ensemble'
        ]
        palette = {
            'Buy & Hold': 'tab:gray',
            'Best Strategy': 'tab:green',
            'Random Strategy': 'tab:red',
            'Fixed Rotation': 'tab:purple',
            'Majority Voting': 'tab:brown',
            'Strategy Ensemble': 'tab:orange'
        }

        for name in benchmark_order:
            curve_data = benchmark_curves.get(name, {})
            curve = curve_data.get('cumulative_returns', [])
            if not curve:
                continue
            curve_pct = [float(v) * 100 for v in curve]
            curve_ts = curve_data.get('market_timestamps', [])
            x_curve, curve_is_datetime = self._resolve_plot_x_axis(curve_ts, len(curve_pct))
            if is_datetime and not curve_is_datetime and len(curve_pct) == len(main_curve):
                x_curve = x_main
            ax.plot(
                x_curve,
                curve_pct,
                linewidth=1.2,
                alpha=0.9,
                label=name,
                color=palette.get(name, None)
            )

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Test Cumulative Return (RL vs Benchmarks)')
        ax.set_xlabel('Timestamp' if is_datetime else 'Trading Step')
        ax.set_ylabel('Cumulative Return (%)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

        if is_datetime:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        return is_datetime
    
    def _generate_backtest_visualization(self, backtest_result: Dict[str, Any], result_name: str):
        """生成回测可视化图表"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            strategy_labels = [
                'MarketNeutral', 'DualMA_MACD', 'BollingerBands',
                'CTA', 'StatArb', 'LongShortEquity'
            ]
            
            # 1. 测试集累计收益率对比曲线（主策略+基准）
            portfolio_values = backtest_result.get('portfolio_values', [])
            self._plot_cumulative_return_with_benchmarks(axes[0, 0], backtest_result)
            
            # 2. 测试集策略选择曲线
            actions_history = backtest_result.get('actions_history', [])
            if actions_history:
                market_timestamps = backtest_result.get('market_timestamps', [])
                x_axis, is_datetime = self._resolve_plot_x_axis(market_timestamps, len(actions_history))
                axes[0, 1].step(x_axis, actions_history, where='post', linewidth=1.0, color='tab:orange')
                axes[0, 1].set_title('Test Strategy Selection Curve')
                axes[0, 1].set_xlabel('Timestamp' if is_datetime else 'Trading Step')
                axes[0, 1].set_ylabel('Strategy')
                axes[0, 1].set_yticks(range(len(strategy_labels)))
                axes[0, 1].set_yticklabels(strategy_labels)
                axes[0, 1].grid(True, alpha=0.3)
                if is_datetime:
                    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    axes[0, 1].xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # 3. 奖励分布
            rewards_history = backtest_result.get('rewards_history', [])
            if rewards_history:
                axes[1, 0].hist(rewards_history, bins=50, alpha=0.7)
                axes[1, 0].axvline(x=np.mean(rewards_history), color='r', linestyle='--', 
                                  label=f'Mean: {np.mean(rewards_history):.3f}')
                axes[1, 0].set_title('Reward Distribution')
                axes[1, 0].set_xlabel('Reward')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 性能指标
            metrics = backtest_result.get('metrics', {})
            if metrics:
                metric_names = ['Total Return', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown']
                metric_values = [
                    metrics.get('total_return', 0) * 100,
                    metrics.get('sharpe_ratio', 0),
                    metrics.get('sortino_ratio', 0),
                    metrics.get('max_drawdown', 0) * 100
                ]
                
                colors = ['green' if v > 0 else 'red' for v in metric_values]
                
                bars = axes[1, 1].bar(metric_names, metric_values, color=colors, alpha=0.7)
                axes[1, 1].set_title('Performance Metrics')
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                # 添加数值标签
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.2f}', ha='center', va='bottom' if value >= 0 else 'top')
            
            plt.tight_layout()
            fig.autofmt_xdate(rotation=30)
            
            # 保存图像
            plot_path = os.path.join(self.output_dir, f"{result_name}_visualization.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"回测可视化图表已保存到: {plot_path}")

            # 单独导出：累计收益率曲线
            if portfolio_values:
                fig1, ax1 = plt.subplots(figsize=(12, 5))
                self._plot_cumulative_return_with_benchmarks(ax1, backtest_result)
                cumret_path = os.path.join(self.output_dir, f"{result_name}_cumulative_return_curve.png")
                fig1.tight_layout()
                fig1.autofmt_xdate(rotation=30)
                fig1.savefig(cumret_path, dpi=300, bbox_inches='tight')
                plt.close(fig1)
                print(f"累计收益率曲线已保存到: {cumret_path}")

            # 单独导出：策略选择曲线
            if actions_history:
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                market_timestamps = backtest_result.get('market_timestamps', [])
                x_axis, is_datetime = self._resolve_plot_x_axis(market_timestamps, len(actions_history))
                ax2.step(x_axis, actions_history, where='post', linewidth=1.0, color='tab:orange')
                ax2.set_title('Test Strategy Selection Curve')
                ax2.set_xlabel('Timestamp' if is_datetime else 'Trading Step')
                ax2.set_ylabel('Strategy')
                ax2.set_yticks(range(len(strategy_labels)))
                ax2.set_yticklabels(strategy_labels)
                ax2.grid(True, alpha=0.3)
                if is_datetime:
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
                strategy_curve_path = os.path.join(self.output_dir, f"{result_name}_strategy_selection_curve.png")
                fig2.tight_layout()
                fig2.autofmt_xdate(rotation=30)
                fig2.savefig(strategy_curve_path, dpi=300, bbox_inches='tight')
                plt.close(fig2)
                print(f"策略选择曲线已保存到: {strategy_curve_path}")
            
        except Exception as e:
            print(f"生成可视化图表时出错: {e}")
    
    def _generate_comparison_visualization(self, comparison: Dict[str, Any], timestamp: str):
        """生成比较可视化图表"""
        try:
            # 这里可以添加比较可视化的实现
            # 例如：多个回测结果的对比柱状图、雷达图等
            pass
        except Exception as e:
            print(f"生成比较可视化图表时出错: {e}")
    
    def _generate_strategy_comparison_visualization(self, 
                                                   comparison: Dict[str, Any], 
                                                   strategy_results: Dict[str, Dict[str, Any]],
                                                   timestamp: str):
        """生成策略对比可视化图表"""
        try:
            # 这里可以添加策略对比可视化的实现
            pass
        except Exception as e:
            print(f"生成策略对比可视化图表时出错: {e}")
    
    def _generate_sensitivity_visualization(self, analysis: Dict[str, Any], timestamp: str):
        """生成敏感性分析可视化图表"""
        try:
            # 这里可以添加敏感性分析可视化的实现
            pass
        except Exception as e:
            print(f"生成敏感性分析可视化图表时出错: {e}")
    
    def get_backtest_results(self) -> Dict[str, Dict[str, Any]]:
        """获取所有回测结果"""
        return self.backtest_results
    
    def get_comparison_results(self) -> Dict[str, Any]:
        """获取比较结果"""
        return self.comparison_results
    
    def clear_results(self):
        """清空所有结果"""
        self.backtest_results = {}
        self.comparison_results = {}
    
    def print_summary(self):
        """打印回测摘要"""
        print("\n" + "="*80)
        print("回测引擎摘要")
        print("="*80)
        print(f"数据路径: {self.data_path}")
        print(f"股票代码: {self.symbols}")
        print(f"初始资金: {self.initial_capital:,.2f}")
        print(f"基础周期: {self.period_minutes}分钟")
        print(f"回测结果数量: {len(self.backtest_results)}")
        print(f"比较结果数量: {len(self.comparison_results)}")
        print("="*80)
    
    def __str__(self) -> str:
        return (f"BacktestEngine(data_path={self.data_path}, symbols={self.symbols}, "
                f"capital={self.initial_capital:,.0f}, results={len(self.backtest_results)})")
    
    def __repr__(self) -> str:
        return self.__str__()
