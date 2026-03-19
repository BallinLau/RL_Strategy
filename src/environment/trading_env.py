"""
交易环境 (Trading Environment)

基于OpenAI Gym的强化学习交易环境
"""

import sys
import hashlib
import pickle
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from src.data.data_preprocessor import DataPreprocessor
from src.features.indicators import IndicatorCalculator
from src.features.indicator_manager import IndicatorManager
from src.features.market_state import MarketStateManager
from src.strategies.strategy_manager import StrategyManager
from .state_space import StateSpace
from .reward_calculator import RewardCalculator
from .position_allocator import PositionAllocator


class TradingEnvironment(gym.Env):
    """
    多策略交易环境
    
    符合OpenAI Gym接口的交易环境
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_capital: float = 1000000,
                 period_minutes: int = 20,
                 state_lookback_days: int = 5,
                 switch_period_multiplier: int = 1,
                 transaction_cost: float = 0.001,
                 dynamic_switch: bool = True,  # 新增：是否启用动态切换
                 state_update_frequency: str = 'dynamic',  # 新增：状态空间更新频率
                 allow_short: bool = True,
                 max_short_per_symbol_ratio: float = 0.1,
                 max_total_short_ratio: float = 0.25,
                 max_position_ratio: float = 0.3,
                 total_position_ratio: float = 0.8,
                 rebalance_tolerance: float = 0.02,
                 max_trade_fraction: float = 0.01,
                 slippage_rate: float = 0.0002,
                 impact_rate: float = 0.0001,
                 min_commission: float = 2.0,
                 min_signal_threshold: float = 0.08,
                 signal_power: float = 1.5,
                 reward_config: dict = None,
                 profit_lock_enabled: bool = False,
                 profit_lock_min_return: float = 0.03,
                 profit_lock_drawdown_threshold: float = 0.08,
                 profit_lock_cooldown_steps: int = 10,
                 profit_lock_safe_action: int = 0,
                 min_action_hold_steps: int = 1):  # 新增：奖励函数配置
        """
        初始化交易环境（增强版）
        
        Args:
            data: 历史数据（分钟级）
            initial_capital: 初始资金
            period_minutes: 基础周期（分钟）
            switch_period_multiplier: 策略切换周期倍数（基础值）
            transaction_cost: 交易成本比例
            dynamic_switch: 是否启用动态切换周期（文档要求的功能）
        """
        super().__init__()
        
        self.data = data
        self.initial_capital = initial_capital
        self.period_minutes = max(int(period_minutes), 1)
        self.state_lookback_days = max(int(state_lookback_days), 1)
        self.base_switch_period = self.period_minutes * switch_period_multiplier
        self.transaction_cost = transaction_cost
        self.dynamic_switch = dynamic_switch
        self.state_update_frequency = state_update_frequency
        self.allow_short = allow_short
        self.max_short_per_symbol_ratio = max_short_per_symbol_ratio
        self.max_total_short_ratio = max_total_short_ratio
        self.max_position_ratio = max_position_ratio
        self.total_position_ratio = total_position_ratio
        self.rebalance_tolerance = rebalance_tolerance
        self.max_trade_fraction = max_trade_fraction
        self.slippage_rate = slippage_rate
        self.impact_rate = impact_rate
        self.min_commission = min_commission
        self.min_signal_threshold = min_signal_threshold
        self.signal_power = signal_power
        # 数值稳定保护阈值，避免过于激进的人为裁剪扭曲训练信号
        self.max_step_value_change_rate = 1.0  # 单步组合价值允许最大变化100%
        self.max_portfolio_multiple = 100.0
        # 锁盈机制：达到一定浮盈后，若从峰值回撤过大，进入防守动作冷却期
        self.profit_lock_enabled = bool(profit_lock_enabled)
        self.profit_lock_min_return = max(float(profit_lock_min_return), 0.0)
        self.profit_lock_drawdown_threshold = max(float(profit_lock_drawdown_threshold), 0.0)
        self.profit_lock_cooldown_steps = max(int(profit_lock_cooldown_steps), 1)
        self.profit_lock_safe_action = int(profit_lock_safe_action)
        self.min_action_hold_steps = max(int(min_action_hold_steps), 1)
        self.peak_portfolio_value = initial_capital
        self.profit_lock_active_until_step = -1
        self.last_profit_lock_trigger_step = -1
        self.last_action_switch_step = 0
        
        # 动态切换周期映射（文档要求：不同市场状态不同切换频率）
        self.switch_period_map = {
            'BULL': 3,      # 牛市：3倍周期（60分钟），不需要频繁切换
            'BEAR': 2,      # 熊市：2倍周期（40分钟）
            'SIDEWAYS': 1,  # 震荡市：1倍周期（20分钟），需要更频繁切换
            'CORRECTION': 1,
            'REBOUND': 1
        }
        
        # 状态更新频率映射（文档要求的功能）
        self.state_update_frequency_map = {
            'high': 1,      # 高频更新：每个周期都更新状态
            'medium': 2,    # 中频更新：每2个周期更新一次状态
            'low': 3,       # 低频更新：每3个周期更新一次状态
            'dynamic': 0    # 动态更新：根据市场状态决定
        }
        
        # 当前切换周期（动态更新）
        self.current_switch_period = self.base_switch_period
        
        # 状态更新计数器
        self.state_update_counter = 0
        self.last_state_update_step = 0
        self.cached_state = None
        
        # 初始化各模块
        self.data_preprocessor = DataPreprocessor(self.period_minutes)
        # 修复缓存路径：确保从项目根目录查找缓存
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cache", "indicators")
        self.indicator_manager = IndicatorManager(self.period_minutes, cache_dir=cache_dir)
        self.market_state_manager = MarketStateManager()
        self.strategy_manager = StrategyManager()
        self.position_allocator = PositionAllocator(
            initial_capital,
            max_position_ratio=max_position_ratio,
            total_position_ratio=total_position_ratio,
            allow_short=allow_short,
            min_signal_threshold=min_signal_threshold,
            signal_power=signal_power
        )
        
        # 初始化奖励计算器（使用配置参数）
        if reward_config:
            self.reward_calculator = RewardCalculator(**reward_config)
        else:
            self.reward_calculator = RewardCalculator()
        
        # 获取股票列表
        self.symbols = self._get_symbols()
        
        # 状态和动作空间
        self.state_space = StateSpace(
            num_stocks=len(self.symbols),
            lookback_window=self.state_lookback_days
        )
        self.action_space = spaces.Discrete(6)  # 6种策略
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_space.get_state_dim(),),
            dtype=np.float32
        )
        
        # 数据若已在预处理阶段聚合（含original_timestamp），则直接使用，避免二次聚合
        if 'original_timestamp' in data.columns:
            print(f"检测到已聚合数据，直接使用（周期配置: {self.period_minutes}分钟）...")
            self.period_data = data.copy()
        else:
            print(f"聚合数据到{self.period_minutes}分钟周期...")
            self.period_data = self.data_preprocessor.aggregate_to_period(data)
        
        # 数据采样：暂时禁用采样以避免时间序列不连续问题
        # sample_ratio = 0.5  # 采样50%的数据点，平衡质量和速度
        # if len(self.period_data) > 5000:  # 只有数据量大时才采样
        #     print(f"数据量较大({len(self.period_data)}行)，采样{sample_ratio*100}%数据点...")
        #     # 均匀采样，保持时间序列特性
        #     sample_indices = np.linspace(0, len(self.period_data)-1, 
        #                                int(len(self.period_data) * sample_ratio), 
        #                                dtype=int)
        #     self.period_data = self.period_data.iloc[sample_indices].reset_index(drop=True)
        #     print(f"采样后数据量: {len(self.period_data)}行")
        
        # 禁用数据采样，使用完整的时间序列数据
        print(f"使用完整数据: {len(self.period_data)}行")
        
        self.max_steps = max(len(self.period_data) - 10, 10)  # 预留10个周期，最少10步
        
        # 预计算所有技术指标（性能优化）
        print("\n" + "="*80)
        print("性能优化：预计算技术指标")
        print("="*80)
        self.precomputed_indicators = self.indicator_manager.precompute_all_indicators(
            self.period_data
        )
        self._build_symbol_indicator_cache()
        print("="*80 + "\n")
        
        # 环境状态
        self.current_step = 0
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: {'quantity': x, 'avg_cost': x}}
        self.current_strategy = None
        self.episode_history = []
        
        print(f"交易环境初始化完成:")
        print(f"  股票数量: {len(self.symbols)}")
        print(f"  数据周期: {self.period_minutes}分钟")
        print(f"  状态窗口: {self.state_lookback_days}天 (t-{self.state_lookback_days-1}..t)")
        print(f"  最大步数: {self.max_steps}")
        print(f"  初始资金: {initial_capital:,.2f}")
        print(f"  预计算指标(主序列): {len(self.precomputed_indicators)} 个时间点")
        print(f"  预计算指标(按股票): {len(getattr(self, 'symbol_indicator_cache', {}))} 只股票")
    
    def reset(self) -> np.ndarray:
        """
        重置环境
        
        Returns:
            observation: 初始观察
        """
        self.current_step = min(10, len(self.period_data) // 4)  # 从较小的步数开始
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.current_strategy = None
        self.episode_history = []
        self.state_space.reset_history()
        self.peak_portfolio_value = self.initial_capital
        self.profit_lock_active_until_step = -1
        self.last_profit_lock_trigger_step = -1
        self.last_action_switch_step = int(self.current_step)
        
        # 重置各模块
        self.reward_calculator.reset()
        self.strategy_manager.reset_all_strategies()
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步（修复版：防止未来信息泄露）
        
        正确的时间序列逻辑：
        1. 基于历史信息（t-1及之前）做决策
        2. 时间推进到t
        3. 用时间t的价格执行交易
        4. 获得时间t的奖励和状态
        
        Args:
            action: 策略选择（0-5）
        
        Returns:
            observation: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 0. 根据交易约束修正动作（避免策略语义与环境约束冲突）
        effective_action = self._sanitize_action_for_constraints(action)
        profit_lock_active, profit_lock_reason = self._resolve_profit_lock_action()
        if profit_lock_active:
            effective_action = self._sanitize_action_for_constraints(self.profit_lock_safe_action)
        else:
            effective_action = self._enforce_min_action_hold(effective_action)

        # 1. 获取当前状态（用于决策，基于历史信息）
        current_state = self._get_raw_state()
        
        # 2. 根据市场状态更新切换周期（恢复动态切换功能）
        self._update_switch_period(current_state)
        
        # 3. 时间推进（恢复动态步进，这是论文的核心创新）
        steps_to_advance = max(1, self.current_switch_period // self.period_minutes)
        self.current_step += steps_to_advance
        
        # 4. 基于历史信息执行策略，但用当前时间的价格交易
        self._execute_strategy_with_current_prices(effective_action, current_state)
        
        # 5. 更新持仓价值（基于新的时间点）
        self._update_portfolio_value()
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        
        # 6. 获取下一个状态（时间推进后的状态）
        next_state = self._get_raw_state()
        observation = self._get_observation()
        
        # 7. 检查是否结束
        done = self.current_step >= self.max_steps

        # 8. 计算奖励（传入done用于终局收益对齐）
        reward = self.reward_calculator.calculate_reward(
            current_state, effective_action, next_state, done=done
        )

        # 9. 收集信息
        peak_drawdown = self._current_drawdown_from_peak()
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'strategy': self.strategy_manager.get_strategy_name(effective_action),
            'raw_action': int(action),
            'effective_action': int(effective_action),
            'step': self.current_step,
            'switch_period': self.current_switch_period,
            'market_state': current_state.get('market_state', {}).get('state', 'SIDEWAYS'),
            'reward_stats': self.reward_calculator.get_statistics(),
            'steps_advanced': steps_to_advance,
            'profit_lock_active': bool(profit_lock_active),
            'profit_lock_reason': profit_lock_reason,
            'profit_lock_active_until_step': int(self.profit_lock_active_until_step),
            'portfolio_peak': float(self.peak_portfolio_value),
            'peak_drawdown': float(peak_drawdown)
        }
        
        # 记录历史
        self.episode_history.append(info)

        return observation, reward, done, info

    def _sanitize_action_for_constraints(self, action: int) -> int:
        """按环境约束修正动作，避免无效策略主导训练。"""
        action = int(action)
        if not self.allow_short:
            # LongShortEquity在禁止做空时信号语义严重失配，退化为稳态基线动作
            if action == 5:
                return 0
        return action

    def _enforce_min_action_hold(self, action: int) -> int:
        """限制策略最短持有步数，降低过度切换。"""
        action = int(action)
        if self.min_action_hold_steps <= 1:
            return action
        if self.current_strategy is None:
            self.last_action_switch_step = int(self.current_step)
            return action
        if action == int(self.current_strategy):
            return action

        held_steps = int(self.current_step - self.last_action_switch_step)
        if held_steps < self.min_action_hold_steps:
            return int(self.current_strategy)

        self.last_action_switch_step = int(self.current_step)
        return action

    def _current_drawdown_from_peak(self) -> float:
        """计算当前相对组合峰值回撤。"""
        peak = max(float(self.peak_portfolio_value), 1e-9)
        drawdown = (peak - float(self.portfolio_value)) / peak
        return max(float(drawdown), 0.0)

    def _resolve_profit_lock_action(self) -> Tuple[bool, str]:
        """
        判断是否触发锁盈防守。

        返回:
            (是否激活锁盈, 原因)
        """
        if not self.profit_lock_enabled:
            return False, 'disabled'

        # 更新峰值
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)

        # 冷却期内持续使用防守动作
        if self.current_step <= self.profit_lock_active_until_step:
            return True, 'cooldown'

        peak_return = (float(self.peak_portfolio_value) - float(self.initial_capital)) / max(float(self.initial_capital), 1e-9)
        drawdown_from_peak = self._current_drawdown_from_peak()
        if peak_return >= self.profit_lock_min_return and drawdown_from_peak >= self.profit_lock_drawdown_threshold:
            self.last_profit_lock_trigger_step = int(self.current_step)
            self.profit_lock_active_until_step = int(self.current_step + self.profit_lock_cooldown_steps)
            return True, 'triggered'

        return False, 'inactive'
    
    def _update_switch_period(self, current_state: Dict[str, Any]):
        """
        根据市场状态更新切换周期（文档要求的功能）
        
        文档要求：不同市场状态下，切换频率可以不同
        牛市不需要频繁切换，可以放长其策略切换周期
        """
        if not self.dynamic_switch:
            self.current_switch_period = self.base_switch_period
            return
        
        # 获取当前市场状态
        market_state = current_state.get('market_state', {}).get('state', 'SIDEWAYS')
        
        # 根据市场状态获取切换周期倍数
        multiplier = self.switch_period_map.get(market_state, 1)
        
        # 基于基础切换周期动态调节，而非直接绑定period_minutes
        self.current_switch_period = self.base_switch_period * multiplier
        
        # 确保切换周期是市场状态周期的倍数（文档要求）
        # 市场状态评估周期是20分钟，所以切换周期应该是20的倍数
        if self.current_switch_period % self.period_minutes != 0:
            self.current_switch_period = self.period_minutes * max(1, int(self.current_switch_period / self.period_minutes))
    
    def _get_observation(self) -> np.ndarray:
        """获取观察（编码后的状态向量）"""
        raw_state = self._get_raw_state()
        return self.state_space.encode_state(raw_state)
    
    def _get_raw_state(self) -> Dict[str, Any]:
        """获取原始状态（优化版 - 使用缓存避免重复计算）"""
        # 检查是否需要更新状态
        if not self._should_update_state():
            return self.cached_state
        
        if self.current_step >= len(self.period_data):
            # 如果超出数据范围，返回最后一个状态
            self.current_step = len(self.period_data) - 1
        
        # 优化：使用预计算的价格缓存
        if not hasattr(self, 'price_cache'):
            self._build_price_cache()
        
        # 1. 价格信息（从缓存获取）
        prices = self.price_cache.get(self.current_step, {})
        
        # 2. 持仓信息
        positions_info = {}
        for symbol, pos in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]['close']
                pnl = (current_price - pos['avg_cost']) * pos['quantity']
                positions_info[symbol] = {
                    'quantity': pos['quantity'],
                    'avg_cost': pos['avg_cost'],
                    'pnl': pnl
                }
        
        # 3. 技术指标（分股票预计算；状态向量仍使用主股票指标以保持维度兼容）
        indicators_by_symbol = {}
        for symbol in self.symbols:
            indicators_by_symbol[symbol] = self._get_symbol_indicators(symbol, self.current_step)
        main_symbol = self.symbols[0] if self.symbols else None
        latest_indicators = indicators_by_symbol.get(main_symbol, {})
        
        # 4. 市场状态：使用全市场代理价格序列（而非固定首只股票）
        proxy_close_series = []
        for step in range(max(0, self.current_step - 20), self.current_step + 1):
            step_prices = self.price_cache.get(step, {})
            step_closes = []
            for px in step_prices.values():
                close_px = px.get('close') if isinstance(px, dict) else None
                if isinstance(close_px, (int, float)) and close_px > 0 and np.isfinite(close_px):
                    step_closes.append(float(close_px))
            if step_closes:
                proxy_close_series.append(float(np.mean(step_closes)))

        if len(proxy_close_series) >= 2:
            market_state = self.market_state_manager.update(pd.Series(proxy_close_series))
        else:
            market_state = {
                'state': 'SIDEWAYS',
                'volatility': 0.0,
                'fast_momentum': 0.0,
                'slow_momentum': 0.0
            }

        # 构建策略代理状态（价格/指标/持仓均做截面聚合）
        proxy_price = {}
        if prices:
            opens, highs, lows, closes, vols = [], [], [], [], []
            for px in prices.values():
                if not isinstance(px, dict):
                    continue
                o = px.get('open', np.nan)
                h = px.get('high', np.nan)
                l = px.get('low', np.nan)
                c = px.get('close', np.nan)
                v = px.get('volume', 0.0)
                if all(isinstance(x, (int, float)) and np.isfinite(x) and x > 0 for x in [o, h, l, c]):
                    opens.append(float(o))
                    highs.append(float(h))
                    lows.append(float(l))
                    closes.append(float(c))
                    vols.append(float(v) if isinstance(v, (int, float)) and np.isfinite(v) else 0.0)
            if closes:
                proxy_price = {
                    'open': float(np.mean(opens)),
                    'high': float(np.mean(highs)),
                    'low': float(np.mean(lows)),
                    'close': float(np.mean(closes)),
                    'volume': float(np.sum(vols))
                }

        proxy_indicators = {}
        if indicators_by_symbol:
            indicator_buckets = {}
            for symbol_map in indicators_by_symbol.values():
                if not isinstance(symbol_map, dict):
                    continue
                for key, value in symbol_map.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        indicator_buckets.setdefault(key, []).append(float(value))
            for key, values in indicator_buckets.items():
                if values:
                    proxy_indicators[key] = float(np.median(values))

        proxy_positions = {'quantity': 0.0, 'avg_cost': 0.0, 'pnl': 0.0}
        if positions_info:
            total_qty = 0.0
            gross_qty = 0.0
            total_cost_weight = 0.0
            total_pnl = 0.0
            for pos in positions_info.values():
                qty = float(pos.get('quantity', 0.0))
                avg_cost = float(pos.get('avg_cost', 0.0))
                pnl = float(pos.get('pnl', 0.0))
                total_qty += qty
                gross_qty += abs(qty)
                total_cost_weight += abs(qty) * avg_cost
                total_pnl += pnl
            proxy_positions['quantity'] = total_qty
            proxy_positions['avg_cost'] = (total_cost_weight / max(gross_qty, 1.0))
            proxy_positions['pnl'] = total_pnl

        # 5. 策略信号
        state_for_strategy = {
            'price': proxy_price,
            'indicators': proxy_indicators,
            'positions': proxy_positions,
            'market_state': market_state,
            'current_time': self.current_step
        }
        all_signals = self.strategy_manager.get_all_signals(state_for_strategy)
        strategy_signals = {}
        for i, sig in all_signals.items():
            signal_strength = sig['signal_strength']
            # 清理策略信号中的nan/inf
            if pd.isna(signal_strength) or np.isinf(signal_strength):
                strategy_signals[i] = 0.0
            else:
                strategy_signals[i] = float(signal_strength)
        
        # 清理市场状态中的nan/inf
        cleaned_market_state = {}
        for key, value in market_state.items():
            if isinstance(value, (int, float)):
                if pd.isna(value) or np.isinf(value):
                    cleaned_market_state[key] = 0.0
                else:
                    cleaned_market_state[key] = float(value)
            else:
                cleaned_market_state[key] = value
        market_state = cleaned_market_state
        
        # 构建完整状态
        state = {
            'prices': prices,
            'positions': positions_info,
            'indicators': proxy_indicators if proxy_indicators else latest_indicators,
            'indicators_by_symbol': indicators_by_symbol,
            'market_state': market_state,
            'strategy_signals': strategy_signals,
            'portfolio_value': self.portfolio_value,
            'current_time': self.current_step
        }
        
        # 缓存状态
        self.cached_state = state
        self.last_state_update_step = self.current_step
        
        return state
    
    def _get_cache_key(self):
        """生成缓存键，基于数据内容和配置参数"""
        # 使用数据的哈希值和关键参数生成缓存键
        data_hash = hashlib.md5(str(self.period_data.shape).encode()).hexdigest()[:8]
        symbols_hash = hashlib.md5(''.join(sorted([str(s) for s in self.symbols])).encode()).hexdigest()[:8]
        period_hash = hashlib.md5(str(self.period_minutes).encode()).hexdigest()[:4]
        
        return f"price_cache_{data_hash}_{symbols_hash}_{period_hash}"
    
    def _save_price_cache(self):
        """保存价格缓存到文件"""
        try:
            cache_dir = "cache/price_cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_key = self._get_cache_key()
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            cache_data = {
                'price_cache': self.price_cache,
                'max_steps': self.max_steps,
                'symbols': self.symbols,
                'period_minutes': self.period_minutes,
                'data_shape': self.period_data.shape
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"价格缓存已保存到: {cache_file}")
            
        except Exception as e:
            print(f"保存价格缓存失败: {e}")
    
    def _load_price_cache(self):
        """从文件加载价格缓存"""
        try:
            cache_dir = "cache/price_cache"
            if not os.path.exists(cache_dir):
                return False
            
            cache_key = self._get_cache_key()
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            if not os.path.exists(cache_file):
                return False
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 验证缓存数据的有效性
            if (cache_data.get('symbols') == self.symbols and
                cache_data.get('period_minutes') == self.period_minutes and
                cache_data.get('data_shape') == self.period_data.shape):
                
                self.price_cache = cache_data['price_cache']
                self.max_steps = cache_data['max_steps']
                
                print(f"价格缓存已从文件加载: {cache_file}")
                print(f"缓存包含 {len(self.price_cache)} 个时间点")
                return True
            else:
                print("缓存数据不匹配，需要重新构建")
                return False
                
        except Exception as e:
            print(f"加载价格缓存失败: {e}")
            return False
    def _build_price_cache(self):
        """预计算价格缓存，避免重复的DataFrame过滤操作（简化版）"""
        print("构建价格缓存...")
        
        # 尝试从缓存文件加载
        if self._load_price_cache():
            return
        
        # 缓存未命中，重新构建
        print("缓存未命中，开始构建新的价格缓存...")
        self.price_cache = {}
        
        # 获取所有唯一时间点，按时间排序
        unique_timestamps = sorted(self.period_data['timestamp'].unique())
        self.unique_timestamps = unique_timestamps
        print(f"发现 {len(unique_timestamps)} 个唯一时间点")
        
        # 按时间点构建价格缓存
        for step, timestamp in enumerate(unique_timestamps):
            step_prices = {}
            
            # 获取当前时间点的所有股票数据
            current_time_data = self.period_data[self.period_data['timestamp'] == timestamp]
            
            for symbol in self.symbols:
                # 查找该股票在当前时间点的数据
                symbol_data = current_time_data[current_time_data['symbol'] == symbol]
                
                if len(symbol_data) > 0:
                    # 找到当前时间点的数据
                    latest = symbol_data.iloc[0]
                    
                    # 简单验证：只检查基本有效性
                    if (not pd.isna(latest['close']) and latest['close'] > 0 and
                        not pd.isna(latest['open']) and latest['open'] > 0 and
                        not pd.isna(latest['high']) and latest['high'] > 0 and
                        not pd.isna(latest['low']) and latest['low'] > 0):
                        
                        step_prices[symbol] = {
                            'open': float(latest['open']),
                            'high': float(latest['high']),
                            'low': float(latest['low']),
                            'close': float(latest['close']),
                            'volume': float(latest['volume']) if not pd.isna(latest['volume']) else 0.0,
                            'timestamp': timestamp
                        }
                    elif step > 0 and symbol in self.price_cache.get(step - 1, {}):
                        # 使用前一步的价格
                        step_prices[symbol] = self.price_cache[step - 1][symbol].copy()
                        step_prices[symbol]['timestamp'] = timestamp
                else:
                    # 当前时间点没有该股票的数据，使用前一步的价格
                    if step > 0 and symbol in self.price_cache.get(step - 1, {}):
                        step_prices[symbol] = self.price_cache[step - 1][symbol].copy()
                        step_prices[symbol]['timestamp'] = timestamp
            
            self.price_cache[step] = step_prices
            
            # 显示进度
            if (step + 1) % 100 == 0:
                progress = (step + 1) / len(unique_timestamps) * 100
                print(f"进度: {progress:.0f}% ({step + 1}/{len(unique_timestamps)})")
        
        # 更新最大步数为实际的时间点数量
        self.max_steps = len(unique_timestamps) - 1
        print(f"价格缓存构建完成，共 {len(self.price_cache)} 个时间点")
        
        # 保存缓存到文件
        self._save_price_cache()

    def _build_symbol_indicator_cache(self):
        """按股票预计算指标，并按统一时间轴对齐"""
        self.symbol_indicator_cache = {}
        if self.period_data.empty or 'timestamp' not in self.period_data.columns:
            return

        timestamps = pd.DatetimeIndex(
            pd.to_datetime(sorted(self.period_data['timestamp'].unique()), errors='coerce')
        ).dropna()
        if len(timestamps) == 0:
            return

        self.unique_timestamps = timestamps
        print(f"构建按股票指标缓存，时间点: {len(timestamps)}")

        for symbol in self.symbols:
            symbol_df = self.period_data[
                self.period_data['symbol'].astype(str) == str(symbol)
            ].copy()
            if symbol_df.empty:
                continue

            symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
            symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'], errors='coerce')
            symbol_df = symbol_df.dropna(subset=['timestamp'])
            if symbol_df.empty:
                continue

            calculator = IndicatorCalculator(symbol_df)
            indicators = calculator.calculate_all()
            indicator_df = pd.DataFrame(indicators)
            indicator_df.insert(0, 'timestamp', symbol_df['timestamp'].values)
            indicator_df = indicator_df.drop_duplicates(subset=['timestamp'], keep='last')
            indicator_df = indicator_df.set_index('timestamp').sort_index()
            indicator_df = indicator_df.reindex(timestamps).ffill().fillna(0.0)
            indicator_df = indicator_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            self.symbol_indicator_cache[symbol] = indicator_df

    def _get_symbol_indicators(self, symbol: str, step: int) -> Dict[str, float]:
        """获取指定股票在指定step的指标值"""
        cache = getattr(self, 'symbol_indicator_cache', {}).get(symbol)
        if cache is None or cache.empty:
            return {}

        idx = min(max(int(step), 0), len(cache) - 1)
        row = cache.iloc[idx]
        result = {}
        for key, value in row.items():
            if pd.isna(value) or np.isinf(value):
                result[key] = 0.0
            else:
                result[key] = float(value)
        return result
    
    def _should_update_state(self) -> bool:
        """
        判断是否需要更新状态（文档要求的功能）
        
        文档要求：状态空间更新频率可以控制
        1. 高频：每个周期都更新
        2. 中频：每2个周期更新一次
        3. 低频：每3个周期更新一次
        4. 动态：根据市场状态决定
        """
        # 如果是第一次调用，必须更新
        if self.cached_state is None:
            return True
        
        # 计算距离上次更新的步数
        steps_since_last_update = self.current_step - self.last_state_update_step
        
        # 获取更新频率
        if self.state_update_frequency == 'dynamic':
            # 动态更新：根据市场状态决定
            update_frequency = self._get_dynamic_update_frequency()
        else:
            # 固定频率更新
            update_frequency = self.state_update_frequency_map.get(
                self.state_update_frequency, 1
            )
        
        # 检查是否需要更新
        return steps_since_last_update >= update_frequency
    
    def _get_dynamic_update_frequency(self) -> int:
        """
        获取动态更新频率（文档要求的功能）
        
        文档要求：不同市场状态下，状态更新频率可以不同
        牛市：低频更新（3个周期）
        熊市：中频更新（2个周期）
        震荡市：高频更新（1个周期）
        """
        if self.cached_state is None:
            return 1
        
        market_state = self.cached_state.get('market_state', {}).get('state', 'SIDEWAYS')
        
        # 根据市场状态决定更新频率
        if market_state == 'BULL':
            return 3  # 牛市：低频更新
        elif market_state == 'BEAR':
            return 2  # 熊市：中频更新
        else:
            return 1  # 震荡市：高频更新
    
    def _execute_strategy(self, action: int):
        """执行选中的策略"""
        raw_state = self._get_raw_state()
        strategy = self.strategy_manager.get_strategy_by_id(action)
        
        if strategy is None:
            return
        
        # 为每只股票生成信号
        signals = {}
        for symbol in self.symbols:
            # 构建单只股票的状态
            symbol_indicators = raw_state.get('indicators_by_symbol', {}).get(
                symbol, raw_state.get('indicators', {})
            )
            stock_state = {
                'price': raw_state['prices'].get(symbol, {}),
                'indicators': symbol_indicators,
                'positions': raw_state['positions'].get(symbol, {}),
                'market_state': raw_state['market_state'],
                'current_time': raw_state.get('current_time', self.current_step)
            }
            signals[symbol] = strategy.generate_signal(stock_state)
        
        # 根据信号分配资金
        target_positions = self.position_allocator.allocate_by_signal_strength(
            signals, action
        )
        
        # 执行交易
        self._rebalance_portfolio(target_positions, raw_state['prices'])
        
        self.current_strategy = action
    
    def _execute_strategy_with_current_prices(self, action: int, historical_state: Dict[str, Any]):
        """
        基于历史信息做决策，但用当前时间的价格执行交易（防止未来泄露）
        
        Args:
            action: 策略选择
            historical_state: 历史状态（用于决策）
        """
        strategy = self.strategy_manager.get_strategy_by_id(action)
        
        if strategy is None:
            return
        
        # 为每只股票生成信号（基于历史信息）
        signals = {}
        for symbol in self.symbols:
            # 构建单只股票的历史状态
            symbol_indicators = historical_state.get('indicators_by_symbol', {}).get(
                symbol, historical_state.get('indicators', {})
            )
            stock_state = {
                'price': historical_state['prices'].get(symbol, {}),
                'indicators': symbol_indicators,
                'positions': historical_state['positions'].get(symbol, {}),
                'market_state': historical_state['market_state'],
                'current_time': historical_state.get('current_time', self.current_step)
            }
            signals[symbol] = strategy.generate_signal(stock_state)
        
        # 根据信号分配资金
        target_positions = self.position_allocator.allocate_by_signal_strength(
            signals, action
        )
        
        # 获取当前时间的价格（用于实际交易）
        current_prices = self.price_cache.get(self.current_step, {})
        
        # 执行交易（用当前价格）
        self._rebalance_portfolio(target_positions, current_prices)
        
        self.current_strategy = action
    
    def _rebalance_portfolio(self, target_positions: Dict[str, float],
                            prices: Dict[str, Dict[str, float]]):
        """
        调整持仓至目标仓位（任务3.5：严格边界检查实现）
        
        实施更严格的交易限制：
        - 降低单次交易资金使用率到1%初始资金
        - 实施更严格的持仓数量限制（单股最多5000股）
        - 增强交易成本计算，提高摩擦成本
        """
        # 检查现金是否异常（静默处理）
        if self.cash < 0 or self.cash > self.initial_capital * 100:
            # 静默处理，不输出警告
            pass
        
        # 同时遍历“目标仓位 + 当前持仓”，确保目标为0时也会触发平仓
        all_symbols = set(target_positions.keys()) | set(self.positions.keys())
        for symbol in all_symbols:
            target_value = float(target_positions.get(symbol, 0.0))
            if symbol not in prices:
                continue
            
            current_price = prices[symbol]['close']
            
            # 简单的价格验证（只检查基本有效性）
            if pd.isna(current_price) or current_price <= 0:
                continue
            
            current_quantity = self.positions.get(symbol, {}).get('quantity', 0)
            # 仅当目标仓位很小且当前无持仓时跳过，避免漏掉“目标为0的平仓”
            if abs(target_value) < 100 and current_quantity == 0:
                continue
                
            target_quantity = int(target_value / current_price)
            
            # 单股持仓上限由配置控制，避免硬编码导致组合长期低暴露
            max_quantity_absolute = 20000
            max_quantity_by_capital = int(
                self.initial_capital * min(max(self.max_position_ratio, 0.01), 1.0) / current_price
            )
            if current_price < 2.0:
                max_quantity_absolute = 8000
            max_allowed = max(1, min(max_quantity_by_capital, max_quantity_absolute))
            
            if abs(target_quantity) > max_allowed:
                sign = 1 if target_quantity > 0 else -1
                target_quantity = sign * max_allowed
            
            current_value = current_quantity * current_price

            # 小幅目标变化不交易，降低换手与摩擦损耗
            if self.portfolio_value > 0:
                value_gap_ratio = abs(target_value - current_value) / self.portfolio_value
                if value_gap_ratio < self.rebalance_tolerance:
                    continue

            quantity_diff = target_quantity - current_quantity
            
            # 单次调仓幅度由max_trade_fraction控制
            max_trade_quantity = max(1, int(self.initial_capital * self.max_trade_fraction / current_price))
            if abs(quantity_diff) > max_trade_quantity:
                sign = 1 if quantity_diff >= 0 else -1
                quantity_diff = sign * max_trade_quantity
            
            if quantity_diff != 0:
                # 执行交易
                trade_value = abs(quantity_diff) * current_price
                
                # 交易成本（保守但不过度惩罚）
                base_cost = trade_value * self.transaction_cost
                slippage_cost = trade_value * self.slippage_rate
                impact_cost = trade_value * self.impact_rate * min(abs(quantity_diff) / 1000, 1.0)
                stamp_tax = trade_value * 0.001 if quantity_diff < 0 else 0  # 卖出时收印花税0.1%
                total_trade_cost = max(base_cost + slippage_cost + impact_cost + stamp_tax, self.min_commission)
                
                if quantity_diff > 0:  # 买入（增加持仓）
                    total_cost = trade_value + total_trade_cost
                    
                    if total_cost <= self.cash:
                        # 检查交易后现金是否合理
                        new_cash = self.cash - total_cost
                        if new_cash < 0:
                            continue
                        
                        self.cash = new_cash
                        self._update_position(symbol, quantity_diff, current_price)
                else:  # 卖出（减少持仓，允许受限做空）
                    current_quantity = self.positions.get(symbol, {}).get('quantity', 0)
                    
                    # 计算卖出后的持仓
                    new_quantity = current_quantity + quantity_diff  # quantity_diff是负数
                    
                    if new_quantity >= 0:
                        # 正常卖出（不涉及做空）
                        cash_increase = trade_value - total_trade_cost
                        if cash_increase > 0:
                            self.cash += cash_increase
                            self._update_position(symbol, quantity_diff, current_price)
                    else:
                        if not self.allow_short:
                            continue

                        # 允许受限做空：限制单股及总空头敞口，避免极端风险
                        max_short_per_symbol = self.initial_capital * self.max_short_per_symbol_ratio
                        max_total_short = self.initial_capital * self.max_total_short_ratio
                        new_symbol_short = abs(new_quantity) * current_price
                        if new_symbol_short > max_short_per_symbol:
                            continue

                        existing_short = 0.0
                        for s, p in self.positions.items():
                            if p.get('quantity', 0) < 0:
                                px = prices.get(s, {}).get('close', p.get('avg_cost', 0))
                                if px > 0:
                                    existing_short += abs(p['quantity']) * px

                        old_symbol_short = abs(min(current_quantity, 0)) * current_price
                        projected_total_short = existing_short - old_symbol_short + new_symbol_short
                        if projected_total_short > max_total_short:
                            continue

                        cash_increase = trade_value - total_trade_cost
                        if cash_increase > 0:
                            self.cash += cash_increase
                            self._update_position(symbol, quantity_diff, current_price)
                    
                    # 现金与持仓有效性会在组合价值更新阶段统一校验
                
                # 静默检查现金异常
                if self.cash < 0 or self.cash > self.initial_capital * 100:
                    # 静默处理，不输出详情
                    pass
    
    def _update_position(self, symbol: str, quantity_change: int, price: float):
        """更新持仓（支持做空，即负持仓）- 增强异常检测版"""
        # 检查输入参数是否合理
        if pd.isna(price) or price <= 0 or price > 1000000:
            print(f"错误: {symbol} 价格异常 {price}，跳过持仓更新")
            return
        
        if abs(quantity_change) > 1000000:
            print(f"错误: {symbol} 数量变化异常 {quantity_change}，跳过持仓更新")
            return
        
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_cost': 0}
        
        old_quantity = self.positions[symbol]['quantity']
        old_cost = self.positions[symbol]['avg_cost']
        new_quantity = old_quantity + quantity_change
        
        # 检查新数量是否合理
        if abs(new_quantity) > 1000000:
            print(f"错误: {symbol} 新持仓数量异常 {new_quantity}，限制交易")
            return
        
        if new_quantity != 0:
            if (old_quantity >= 0 and quantity_change > 0) or (old_quantity < 0 and quantity_change < 0):
                # 同方向交易：买入更多多头或卖出更多空头，更新平均成本
                if old_quantity == 0:
                    new_avg_cost = price
                else:
                    total_cost = old_quantity * old_cost + quantity_change * price
                    new_avg_cost = total_cost / new_quantity
                    
                    # 检查新平均成本是否合理
                    if pd.isna(new_avg_cost) or new_avg_cost <= 0 or new_avg_cost > 1000000:
                        print(f"错误: {symbol} 平均成本异常 {new_avg_cost}，使用当前价格")
                        new_avg_cost = price
            else:
                # 反方向交易：平仓或转换方向
                if abs(quantity_change) <= abs(old_quantity):
                    # 部分平仓，保持原平均成本
                    new_avg_cost = old_cost if old_cost > 0 else price
                else:
                    # 完全平仓并反向开仓，新的平均成本就是当前价格
                    new_avg_cost = price
            
            # 最终检查：确保平均成本合理
            if pd.isna(new_avg_cost) or new_avg_cost <= 0:
                new_avg_cost = price
            
            self.positions[symbol] = {
                'quantity': new_quantity,
                'avg_cost': new_avg_cost
            }
            
            # 检查持仓价值是否合理
            position_value = abs(new_quantity) * new_avg_cost
            if position_value > self.initial_capital * 0.5:  # 单只股票持仓价值不应超过初始资金的50%
                print(f"警告: {symbol} 持仓价值过大 {position_value:,.0f}，数量: {new_quantity}, 成本: {new_avg_cost}")
        else:
            # 完全平仓
            if symbol in self.positions:
                del self.positions[symbol]
    
    def _update_portfolio_value(self):
        """
        更新组合价值（带数值稳定保护）

        仅在异常数值场景下触发保护，尽量避免强行裁剪导致收益轨迹失真。
        """
        if self.current_step >= len(self.price_cache):
            return
        
        position_value = 0
        
        # 使用价格缓存获取当前价格
        current_prices = self.price_cache.get(self.current_step, {})
        
        for symbol, pos in self.positions.items():
            if pos['quantity'] == 0:
                continue
                
            # 从价格缓存获取当前价格
            if symbol in current_prices:
                current_price = current_prices[symbol].get('close', pos['avg_cost'])
                
                # 检查价格是否合理
                if (pd.isna(current_price) or current_price <= 0 or 
                    current_price > 1000000):  # 价格异常检查
                    current_price = pos['avg_cost']  # 使用平均成本作为备用
                
                stock_value = pos['quantity'] * current_price
                
                # 任务3.2：持仓价值合理性验证（单股不超过初始资金50%）
                max_single_position_value = self.initial_capital * 0.5
                if abs(stock_value) > max_single_position_value:
                    print(f"数值稳定性保护: {symbol} 单股持仓价值 {abs(stock_value):,.0f} 超过限制 {max_single_position_value:,.0f}")
                    # 限制单股持仓价值
                    sign = 1 if stock_value >= 0 else -1
                    stock_value = sign * max_single_position_value
                    print(f"  限制后持仓价值: {stock_value:,.0f}")
                
                position_value += stock_value
            else:
                # 如果价格缓存中没有该股票，使用平均成本
                if pos['avg_cost'] > 0:
                    stock_value = pos['quantity'] * pos['avg_cost']
                    
                    # 任务3.2：对使用平均成本的持仓也进行价值验证
                    max_single_position_value = self.initial_capital * 0.5
                    if abs(stock_value) > max_single_position_value:
                        sign = 1 if stock_value >= 0 else -1
                        stock_value = sign * max_single_position_value
                    
                    position_value += stock_value
        
        # 记录更新前的组合价值
        prev_portfolio_value = self.portfolio_value
        
        # 计算总组合价值
        total_value = self.cash + position_value
        
        # 增强的组合价值异常检测和修复
        if (pd.isna(total_value) or np.isinf(total_value) or total_value < 0):
            print(f"数值稳定性保护: 组合价值计算错误 {total_value:,.0f}")
            print(f"  现金: {self.cash:,.0f}")
            print(f"  持仓价值: {position_value:,.0f}")
            print(f"  持仓详情: {[(s, p['quantity'], p['avg_cost']) for s, p in self.positions.items() if p['quantity'] != 0]}")
            
            # 只在明显错误时重置（NaN、Inf、负值）
            self.portfolio_value = self.initial_capital
            self.cash = self.initial_capital
            self.positions = {}
            return
        
        # 单步异常保护：仅当变化超过配置阈值时才进行软限制
        if prev_portfolio_value > 0:
            value_change_rate = (total_value - prev_portfolio_value) / prev_portfolio_value
            if abs(value_change_rate) > self.max_step_value_change_rate:
                capped_change = np.sign(value_change_rate) * self.max_step_value_change_rate
                total_value = prev_portfolio_value * (1.0 + capped_change)

        # 组合价值绝对限制：仅防止数值爆炸，不设置硬下限避免人为抬升净值
        max_portfolio_value = self.initial_capital * self.max_portfolio_multiple
        min_portfolio_value = 0.0

        if total_value > max_portfolio_value:
            total_value = max_portfolio_value
        elif total_value < min_portfolio_value:
            total_value = min_portfolio_value
        
        self.portfolio_value = total_value
        
        # 记录高收益率情况（但不重置）
        return_rate = (total_value - self.initial_capital) / self.initial_capital
        if abs(return_rate) > 1.0:  # 收益率超过100%时记录
            print(f"数值稳定性保护: 高收益率 {return_rate*100:.1f}% - 组合价值: {total_value:,.0f}, 现金: {self.cash:,.0f}")
            # 不重置，让系统继续运行
        
        # 更新资金分配器的总资金（恢复正常更新）
        self.position_allocator.update_total_capital(self.portfolio_value)
    
    def _get_symbols(self) -> List[str]:
        """获取所有股票代码"""
        if 'symbol' in self.data.columns:
            return self.data['symbol'].unique().tolist()
        return []
    
    def render(self, mode='human'):
        """渲染环境（可选）"""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Portfolio Value: {self.portfolio_value:,.2f}")
            print(f"Cash: {self.cash:,.2f}")
            print(f"Current Strategy: {self.current_strategy}")
            print(f"Positions: {len(self.positions)}")
            for symbol, pos in self.positions.items():
                print(f"  {symbol}: {pos['quantity']} @ {pos['avg_cost']:.2f}")
            print(f"{'='*60}")
    
    def close(self):
        """关闭环境"""
        pass
    
    def get_episode_history(self) -> List[Dict]:
        """获取episode历史"""
        return self.episode_history
    
    def __str__(self) -> str:
        return (f"TradingEnvironment(symbols={len(self.symbols)}, "
                f"capital={self.initial_capital:,.0f}, "
                f"period={self.period_minutes}min)")
    
    def __repr__(self) -> str:
        return self.__str__()
