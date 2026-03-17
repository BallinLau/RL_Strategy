"""
增强版奖励计算器 (Enhanced Reward Calculator)

包含文档中所有功能的完整实现：
1. 策略历史收益率记录
2. 综合评分公式（基础评分 + 历史表现）
3. 截断值机制
4. 完整的参数配置
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque


class RewardCalculator:
    """
    奖励计算器
    
    总奖励 = 收益奖励 + 风险调整奖励 + 切换惩罚 + 行为一致性奖励
    """
    
    def __init__(self, 
                 alpha: float = 1.0,      # 收益奖励系数
                 kappa: float = 0.4,      # 超额收益奖励系数（相对等权买入持有）
                 beta: float = 0.5,       # 波动率惩罚系数
                 gamma: float = 0.3,      # 回撤惩罚系数
                 delta: float = 0.1,      # 策略切换惩罚系数
                 eta: float = 0.2,        # 市场适配性奖励系数
                 theta: float = 0.15,     # 策略稳定性奖励系数
                 zeta: float = 0.1,       # 多样性奖励系数（新增）
                 
                 # 新增参数（文档中提到的功能）
                 history_window: int = 100,      # 历史记录窗口大小
                 performance_weight: float = 0.3, # 历史表现权重 (w2)
                 cutoff_threshold: float = 0.7,  # 适配性评分截断值
                 base_weight: float = 0.7,       # 基础评分权重 (w1)
                 
                 # 回撤参数
                 drawdown_threshold: float = 0.05,  # 回撤阈值 (5%)
                 
                 # 波动率参数
                 volatility_lookback: int = 20,    # 波动率计算回看周期
                 
                 # 交易成本参数（文档要求的功能）
                 commission_rate: float = 0.0003,  # 佣金率 (0.03%)
                 slippage_rate: float = 0.0002,    # 滑点率 (0.02%)
                 market_impact_factor: float = 0.0001,  # 市场冲击系数
                 min_commission: float = 5.0,      # 最低佣金 (元)
                 tax_rate: float = 0.001,         # 印花税率 (0.1%)
                 negative_return_penalty: float = 0.12,
                 terminal_return_bonus: float = 0.8,
                 preserve_history_across_episodes: bool = False,
                 reward_mode: str = 'legacy',
                 excess_reward_weight: float = 1.0,
                 drawdown_penalty_weight: float = 0.8,
                 turnover_penalty_weight: float = 0.06,
                 switch_penalty_weight: float = 1.0,
                 directional_hit_weight: float = 0.0,
                 directional_min_abs_return: float = 0.0005):
        """
        初始化增强版奖励计算器
        
        Args:
            alpha: 收益奖励系数
            kappa: 超额收益奖励系数（相对等权买入持有）
            beta: 波动率惩罚系数
            gamma: 回撤惩罚系数
            delta: 策略切换惩罚系数
            eta: 市场适配性奖励系数
            theta: 策略稳定性惩罚系数
            zeta: 多样性奖励系数（新增）
            history_window: 历史记录窗口大小
            performance_weight: 历史表现权重 (w2)
            cutoff_threshold: 适配性评分截断值
            base_weight: 基础评分权重 (w1)
            drawdown_threshold: 回撤阈值
            volatility_lookback: 波动率计算回看周期
        """
        # 基础参数
        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.theta = theta
        self.zeta = zeta  # 多样性奖励系数
        
        # 新增参数
        self.history_window = history_window
        self.performance_weight = performance_weight
        self.cutoff_threshold = cutoff_threshold
        self.base_weight = base_weight
        self.drawdown_threshold = drawdown_threshold
        self.volatility_lookback = volatility_lookback
        
        # 交易成本参数（文档要求的功能）
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.market_impact_factor = market_impact_factor
        self.min_commission = min_commission
        self.tax_rate = tax_rate
        self.negative_return_penalty = max(float(negative_return_penalty), 0.0)
        self.terminal_return_bonus = float(terminal_return_bonus)
        self.preserve_history_across_episodes = bool(preserve_history_across_episodes)
        self.reward_mode = str(reward_mode).lower()
        self.excess_reward_weight = max(float(excess_reward_weight), 0.0)
        self.drawdown_penalty_weight = max(float(drawdown_penalty_weight), 0.0)
        self.turnover_penalty_weight = max(float(turnover_penalty_weight), 0.0)
        self.switch_penalty_weight = max(float(switch_penalty_weight), 0.0)
        self.directional_hit_weight = max(float(directional_hit_weight), 0.0)
        self.directional_min_abs_return = max(float(directional_min_abs_return), 0.0)
        
        # 内部状态
        self.prev_portfolio_value = None
        self.prev_strategy = None
        self.strategy_hold_periods = 0
        self.daily_returns = []
        self.peak_value = 0
        self.benchmark_value = None
        self.last_directional_hit_rate = 0.5
        self.last_directional_exposure_ratio = 0.0
        
        # 交易成本统计（文档要求的功能）
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_tax = 0.0
        self.total_market_impact = 0.0
        self.trade_count = 0
        
        # 策略历史收益率记录（文档要求的功能）
        self.strategy_performance = {i: deque(maxlen=history_window) for i in range(6)}
        self.strategy_returns = {i: [] for i in range(6)}
        
        # 策略使用统计（用于多样性奖励）
        self.strategy_usage_count = {i: 0 for i in range(6)}
        self.recent_strategies = deque(maxlen=50)  # 记录最近50个决策的策略选择
        
        # 市场适配性评分表（静态）
        self._init_adaptation_matrix()
        
        # 策略基础评分（用于综合评分）
        self.strategy_base_scores = {i: 0.5 for i in range(6)}
    
    def _init_adaptation_matrix(self):
        """初始化策略-市场状态适配评分表"""
        self.adaptation_matrix = {
            # 市场中性策略
            0: {'BULL': 0.5, 'BEAR': 0.5, 'CORRECTION': 0.7, 
                'REBOUND': 0.7, 'SIDEWAYS': 0.9},
            # 双均线/MACD策略
            1: {'BULL': 0.9, 'BEAR': 0.9, 'CORRECTION': 0.6, 
                'REBOUND': 0.6, 'SIDEWAYS': 0.3},
            # 布林带策略
            2: {'BULL': 0.4, 'BEAR': 0.4, 'CORRECTION': 0.7, 
                'REBOUND': 0.7, 'SIDEWAYS': 0.8},
            # CTA策略
            3: {'BULL': 0.95, 'BEAR': 0.95, 'CORRECTION': 0.5, 
                'REBOUND': 0.5, 'SIDEWAYS': 0.2},
            # 统计套利策略
            4: {'BULL': 0.5, 'BEAR': 0.5, 'CORRECTION': 0.8, 
                'REBOUND': 0.8, 'SIDEWAYS': 0.9},
            # 股票多空策略
            5: {'BULL': 0.8, 'BEAR': 0.8, 'CORRECTION': 0.7, 
                'REBOUND': 0.7, 'SIDEWAYS': 0.5}
        }
    
    def calculate_reward(self,
                        current_state: Dict[str, Any],
                        action: int,
                        next_state: Dict[str, Any],
                        done: bool = False) -> float:
        """
        计算奖励（增强版 - 数值稳定与风险约束）
        
        说明：
        - excess_dominant 模式以“超额收益 + 回撤 + 换手(+切换)”为主
        - 不再使用“单步异常收益/累计收益超过阈值”的硬惩罚
        - 保留NaN/Inf防护和合理范围裁剪，避免数值不稳定
        
        Args:
            current_state: 当前状态
            action: 执行的动作（策略ID）
            next_state: 下一个状态
            done: 是否episode结束
        
        Returns:
            total_reward: 总奖励
        """
        current_value = current_state.get('portfolio_value', 0)
        next_value = next_state.get('portfolio_value', 0)
        
        # 增强数值稳定性检查 - 输入验证
        if not isinstance(current_value, (int, float)) or not isinstance(next_value, (int, float)):
            return 0.0
        if np.isnan(current_value) or np.isinf(current_value) or np.isnan(next_value) or np.isinf(next_value):
            return 0.0
        if current_value < 0 or next_value < 0:
            return 0.0
        
        # 初始化
        if self.prev_portfolio_value is None:
            self.prev_portfolio_value = current_value
            self.peak_value = current_value
            self.initial_value = current_value  # 记录初始价值用于基准比较
        
        # 1. 收益奖励（加强数值稳定性和异常收益率检测）
        if self.prev_portfolio_value is None or self.prev_portfolio_value <= 0:
            log_return = 0
        else:
            # 防止除零和极端值
            ratio = next_value / self.prev_portfolio_value
            ratio = max(0.01, min(100, ratio))  # 限制比率在0.01到100之间
            log_return = np.log(ratio)
        
        # 更严格的裁剪，防止极端值（针对时间跳跃问题进一步收紧）
        log_return = np.clip(log_return, -0.05, 0.05)  # 限制到±5%，防止时间跳跃导致的收益率爆炸
        
        # 计算单步收益率和累计收益率
        single_step_return = 0
        if self.prev_portfolio_value > 0:
            single_step_return = (next_value - self.prev_portfolio_value) / self.prev_portfolio_value
        
        # 计算累计收益率
        cumulative_return = 0
        if hasattr(self, 'initial_value') and self.initial_value > 0:
            cumulative_return = (next_value - self.initial_value) / self.initial_value
        
        # 基础收益奖励（PnL主导）
        return_reward = self.alpha * log_return

        # 超额收益奖励：对比当前可交易股票等权买入持有的单步收益
        benchmark_step_return = 0.0
        current_prices = current_state.get('prices', {}) if isinstance(current_state, dict) else {}
        next_prices = next_state.get('prices', {}) if isinstance(next_state, dict) else {}
        if isinstance(current_prices, dict) and isinstance(next_prices, dict):
            benchmark_returns = []
            for symbol, cur_px in current_prices.items():
                if symbol not in next_prices:
                    continue
                c = cur_px.get('close', 0) if isinstance(cur_px, dict) else 0
                n = next_prices[symbol].get('close', 0) if isinstance(next_prices[symbol], dict) else 0
                if isinstance(c, (int, float)) and isinstance(n, (int, float)):
                    if c > 0 and np.isfinite(c) and np.isfinite(n):
                        benchmark_returns.append(np.clip((n - c) / c, -0.2, 0.2))
            if benchmark_returns:
                benchmark_step_return = float(np.mean(benchmark_returns))

        excess_step_return = single_step_return - benchmark_step_return
        excess_reward = self.kappa * float(np.clip(excess_step_return, -0.05, 0.05))

        if self.reward_mode == 'excess_dominant':
            step_excess = float(np.clip(excess_step_return, -0.2, 0.2))
            self._record_strategy_return(action, step_excess)

            if self.benchmark_value is None or not np.isfinite(self.benchmark_value):
                self.benchmark_value = float(max(self.initial_value, 1e-9))
            benchmark_step_return_capped = float(np.clip(benchmark_step_return, -0.2, 0.2))
            self.benchmark_value = float(self.benchmark_value * (1.0 + benchmark_step_return_capped))

            self.peak_value = max(self.peak_value, next_value)
            drawdown = (self.peak_value - next_value) / self.peak_value if self.peak_value > 0 else 0.0
            if not np.isfinite(drawdown):
                drawdown = 0.0
            drawdown_penalty = self.drawdown_penalty_weight * max(drawdown - self.drawdown_threshold, 0.0)

            turnover_ratio = self._estimate_turnover_ratio(current_state, next_state, next_value)
            turnover_penalty = self.turnover_penalty_weight * turnover_ratio

            # 方向命中率与仓位方向绑定：
            # long在上涨时命中、short在下跌时命中；并按仓位绝对市值加权。
            directional_hit_rate, directional_exposure_ratio = self._calculate_directional_hit_rate(
                current_state, next_state
            )
            self.last_directional_hit_rate = directional_hit_rate
            self.last_directional_exposure_ratio = directional_exposure_ratio
            directional_edge = float(np.clip(2.0 * directional_hit_rate - 1.0, -1.0, 1.0))
            directional_reward = (
                self.directional_hit_weight
                * directional_edge
                * min(float(directional_exposure_ratio), 1.0)
            )

            switch_penalty = 0.0
            if self.prev_strategy is not None and action != self.prev_strategy:
                switch_penalty = self.switch_penalty_weight * abs(float(self.delta))

            total_reward = (
                self.excess_reward_weight * step_excess
                - drawdown_penalty
                - turnover_penalty
                - switch_penalty
                + directional_reward
            )

            if done and self.initial_value > 0:
                total_return = (next_value - self.initial_value) / self.initial_value
                benchmark_total_return = (self.benchmark_value - self.initial_value) / self.initial_value
                terminal_excess = float(np.clip(total_return - benchmark_total_return, -1.0, 1.0))
                total_reward += self.terminal_return_bonus * terminal_excess

            if np.isnan(total_reward) or np.isinf(total_reward):
                total_reward = 0.0

            total_reward = np.clip(total_reward, -1.0, 1.0)
            self.prev_portfolio_value = next_value
            self.prev_strategy = action
            if not hasattr(self, 'step_count'):
                self.step_count = 0
            self.step_count += 1
            return float(total_reward)

        # legacy 模式仍记录原始收益轨迹
        self._record_strategy_return(action, log_return)
        
        # 相对收益奖励：奖励超越买入持有策略的表现
        total_return = 0.0
        relative_reward = 0.0
        if hasattr(self, 'initial_value') and self.initial_value > 0:
            # 计算当前总收益率
            total_return = (next_value - self.initial_value) / self.initial_value
            # 相对收益仅作弱引导，避免压过真实PnL信号
            relative_reward = 0.02 * np.clip(total_return, -0.5, 0.5)
        
        # 负收益惩罚：以单步下跌为主，累计亏损只做轻微约束，避免过度惩罚
        downside_penalty = 0.0
        if single_step_return < 0:
            downside_penalty -= self.negative_return_penalty * min(abs(single_step_return), 0.02)
        if total_return < 0:
            downside_penalty -= 0.02 * self.negative_return_penalty * min(abs(total_return), 0.5)
        
        
        # 交易成本已在环境资金流中显式扣除，这里不再重复扣除，避免奖励双重惩罚
        transaction_cost_penalty = 0.0
        
        # 2. 风险调整奖励
        # 2.1 日内波动率（使用最近N个周期的收益率标准差）
        self.daily_returns.append(log_return)
        if len(self.daily_returns) > self.volatility_lookback:
            self.daily_returns.pop(0)
        
        volatility = np.std(self.daily_returns) if len(self.daily_returns) > 1 else 0
        # 数值稳定性检查
        if np.isnan(volatility) or np.isinf(volatility):
            volatility = 0
        volatility_penalty = -self.beta * volatility
        
        # 2.2 回撤惩罚（使用配置的回撤阈值）
        self.peak_value = max(self.peak_value, next_value)
        drawdown = (self.peak_value - next_value) / self.peak_value if self.peak_value > 0 else 0
        # 数值稳定性检查
        if np.isnan(drawdown) or np.isinf(drawdown):
            drawdown = 0
        drawdown_penalty = 0
        if drawdown > self.drawdown_threshold:
            drawdown_penalty = -self.gamma * (drawdown - self.drawdown_threshold)
        
        risk_adjusted_reward = volatility_penalty + drawdown_penalty
        
        # 3. 策略切换惩罚（修正逻辑）
        switch_penalty = 0
        if self.prev_strategy is not None and action != self.prev_strategy:
            # 切换策略给予惩罚（delta是负值）
            switch_penalty = self.delta  # delta=-0.001，所以这是-0.001的惩罚
            self.strategy_hold_periods = 0
        else:
            self.strategy_hold_periods += 1
        
        # 4. 行为一致性奖励（修复版）
        # 4.1 市场适配性奖励（改为相对奖励）
        market_state = next_state.get('market_state', {}).get('state', 'SIDEWAYS')
        adaptation_score = self._calculate_enhanced_adaptation_score(action, market_state)
        
        # 应用截断值（文档要求的功能）
        adaptation_score = min(adaptation_score, self.cutoff_threshold)
        
        # 改为相对奖励：与平均适配性(0.5)比较
        relative_adaptation = adaptation_score - 0.5
        adaptation_reward = self.eta * relative_adaptation
        
        # 4.2 策略多样性奖励（修复：添加系数控制）
        diversity_score = self._calculate_diversity_reward()
        # 改为相对奖励：与平均多样性(0.5)比较，并添加系数控制
        relative_diversity = diversity_score - 0.5
        diversity_reward = self.zeta * relative_diversity
        
        # 4.3 策略稳定性惩罚（修复命名和逻辑）
        stability_penalty = self.theta * min(self.strategy_hold_periods / 10, 1.0)
        
        # 行为项只在“有真实收益波动”时生效，避免出现空仓高奖励
        activity_scale = min(abs(single_step_return) / 0.005, 1.0)  # 单步0.5%及以上视为满激活
        behavior_reward = (adaptation_reward + diversity_reward + stability_penalty) * 0.1 * activity_scale

        # 终局奖励：将episode目标和最终累计收益率显式对齐
        terminal_reward = 0.0
        if done:
            terminal_reward = self.terminal_return_bonus * float(np.clip(total_return, -1.0, 1.0))
        
        # 总奖励
        total_reward = (
            return_reward
            + excess_reward
            + relative_reward
            + downside_penalty
            + risk_adjusted_reward
            + switch_penalty
            + behavior_reward
            + terminal_reward
            + transaction_cost_penalty
        )
        
        # 简单的数值稳定性检查
        if np.isnan(total_reward) or np.isinf(total_reward):
            total_reward = 0.0
        
        # 基本奖励裁剪
        total_reward = np.clip(total_reward, -1.0, 1.0)
        
        # 更新状态
        self.prev_portfolio_value = next_value
        self.prev_strategy = action
        
        # 更新步骤计数器
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1
        
        return float(total_reward)
    
    def _record_strategy_return(self, strategy_id: int, return_value: float):
        """记录策略收益率（文档要求的功能）"""
        self.strategy_performance[strategy_id].append(return_value)
        self.strategy_returns[strategy_id].append(return_value)
        
        # 更新策略使用统计
        self.strategy_usage_count[strategy_id] += 1
        self.recent_strategies.append(strategy_id)

    def _estimate_turnover_ratio(self,
                                 current_state: Dict[str, Any],
                                 next_state: Dict[str, Any],
                                 next_portfolio_value: float) -> float:
        """根据相邻状态持仓变化估算换手率。"""
        try:
            current_positions = current_state.get('positions', {}) if isinstance(current_state, dict) else {}
            next_positions = next_state.get('positions', {}) if isinstance(next_state, dict) else {}
            current_prices = current_state.get('prices', {}) if isinstance(current_state, dict) else {}
            next_prices = next_state.get('prices', {}) if isinstance(next_state, dict) else {}

            traded_notional = 0.0
            symbols = set(current_positions.keys()) | set(next_positions.keys())
            for symbol in symbols:
                cur_pos = current_positions.get(symbol, {}) if isinstance(current_positions, dict) else {}
                nxt_pos = next_positions.get(symbol, {}) if isinstance(next_positions, dict) else {}
                cur_qty = float(cur_pos.get('quantity', 0.0)) if isinstance(cur_pos, dict) else 0.0
                nxt_qty = float(nxt_pos.get('quantity', 0.0)) if isinstance(nxt_pos, dict) else 0.0
                qty_diff = abs(nxt_qty - cur_qty)
                if qty_diff <= 0.0:
                    continue

                price = 0.0
                nxt_px = next_prices.get(symbol, {}) if isinstance(next_prices, dict) else {}
                cur_px = current_prices.get(symbol, {}) if isinstance(current_prices, dict) else {}
                if isinstance(nxt_px, dict):
                    price = float(nxt_px.get('close', 0.0))
                if (not np.isfinite(price)) or price <= 0.0:
                    if isinstance(cur_px, dict):
                        price = float(cur_px.get('close', 0.0))
                if (not np.isfinite(price)) or price <= 0.0:
                    continue

                traded_notional += qty_diff * price

            denominator = max(float(next_portfolio_value), 1.0)
            turnover_ratio = traded_notional / denominator
            if not np.isfinite(turnover_ratio):
                return 0.0
            return float(np.clip(turnover_ratio, 0.0, 5.0))
        except Exception:
            return 0.0

    def _calculate_directional_hit_rate(self,
                                        current_state: Dict[str, Any],
                                        next_state: Dict[str, Any]) -> Tuple[float, float]:
        """
        计算方向命中率（long/short绑定）:
        - long 持仓且下一步收益>0 记命中
        - short 持仓且下一步收益<0 记命中
        采用仓位绝对市值加权，并返回暴露比例用于抑制低仓位噪声。
        """
        try:
            positions = current_state.get('positions', {}) if isinstance(current_state, dict) else {}
            current_prices = current_state.get('prices', {}) if isinstance(current_state, dict) else {}
            next_prices = next_state.get('prices', {}) if isinstance(next_state, dict) else {}
            portfolio_value = float(current_state.get('portfolio_value', 0.0)) if isinstance(current_state, dict) else 0.0

            weighted_hits = 0.0
            gross_exposure = 0.0
            min_abs_ret = float(self.directional_min_abs_return)

            for symbol, pos in positions.items():
                if not isinstance(pos, dict):
                    continue
                qty = float(pos.get('quantity', 0.0))
                if (not np.isfinite(qty)) or abs(qty) <= 0.0:
                    continue

                cur_px_row = current_prices.get(symbol, {}) if isinstance(current_prices, dict) else {}
                nxt_px_row = next_prices.get(symbol, {}) if isinstance(next_prices, dict) else {}
                if not isinstance(cur_px_row, dict) or not isinstance(nxt_px_row, dict):
                    continue

                cur_px = float(cur_px_row.get('close', 0.0))
                nxt_px = float(nxt_px_row.get('close', 0.0))
                if (not np.isfinite(cur_px)) or (not np.isfinite(nxt_px)) or cur_px <= 0.0 or nxt_px <= 0.0:
                    continue

                step_ret = (nxt_px - cur_px) / cur_px
                if (not np.isfinite(step_ret)) or abs(step_ret) < min_abs_ret:
                    continue

                notional = abs(qty) * cur_px
                if (not np.isfinite(notional)) or notional <= 0.0:
                    continue

                gross_exposure += notional
                if qty * step_ret > 0:
                    weighted_hits += notional

            if gross_exposure <= 0.0:
                return 0.5, 0.0

            hit_rate = float(np.clip(weighted_hits / gross_exposure, 0.0, 1.0))
            exposure_ratio = 0.0
            if np.isfinite(portfolio_value) and portfolio_value > 0.0:
                exposure_ratio = float(np.clip(gross_exposure / portfolio_value, 0.0, 2.0))
            return hit_rate, exposure_ratio
        except Exception:
            return 0.5, 0.0
    
    def _calculate_diversity_reward(self) -> float:
        """
        计算策略多样性奖励（修复版）
        
        奖励机制：
        1. 使用更多不同策略 -> 更高奖励
        2. 策略分布越均匀 -> 更高奖励
        3. 避免过度集中在单一策略
        
        Returns:
            diversity_reward: 多样性奖励 (0-1)，平均值约0.5
        """
        if len(self.recent_strategies) < 10:
            return 0.5  # 返回中性值，避免初期偏差
        
        # 计算最近策略的多样性
        recent_list = list(self.recent_strategies)
        unique_strategies = len(set(recent_list))
        
        # 基础多样性分数：使用的不同策略数量 / 总策略数量
        diversity_score = unique_strategies / 6.0
        
        # 分布均匀性奖励：计算策略分布的熵
        strategy_counts = {}
        for s in recent_list:
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        
        # 计算概率分布
        total = len(recent_list)
        probabilities = [strategy_counts.get(i, 0) / total for i in range(6)]
        
        # 计算熵（越高越均匀）
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log(p)
        
        # 最大熵为log(6)，归一化到0-1
        max_entropy = np.log(6)
        entropy_score = entropy / max_entropy if max_entropy > 0 else 0
        
        # 综合多样性奖励（确保在0-1范围）
        diversity_reward = 0.5 * diversity_score + 0.5 * entropy_score
        diversity_reward = np.clip(diversity_reward, 0.0, 1.0)
        
        return diversity_reward
    
    def _calculate_adaptation_score(self, strategy_id: int, market_state: str) -> float:
        """
        计算市场适配性评分
        
        Args:
            strategy_id: 策略ID
            market_state: 市场状态
        
        Returns:
            score: 适配性评分 (0-1)
        """
        return self.adaptation_matrix.get(strategy_id, {}).get(market_state, 0.5)
    
    def _calculate_enhanced_adaptation_score(self, strategy_id: int, market_state: str) -> float:
        """
        计算增强版市场适配性评分（文档要求的功能）
        
        公式：综合评分 = 基础评分 × w1 + 历史平均收益率 × w2
        
        Args:
            strategy_id: 策略ID
            market_state: 市场状态
        
        Returns:
            score: 综合适配性评分 (0-1)
        """
        # 基础评分
        base_score = self.adaptation_matrix.get(strategy_id, {}).get(market_state, 0.5)
        
        # 历史平均收益率（归一化到0-1）
        history_returns = list(self.strategy_performance[strategy_id])
        if len(history_returns) > 0:
            avg_return = np.mean(history_returns)
            # 将收益率映射到0-1范围（假设年化收益率在-0.5到0.5之间）
            normalized_return = (avg_return * 252 + 0.5) / 1.0  # 年化并归一化
            normalized_return = max(0, min(1, normalized_return))  # 截断到[0,1]
        else:
            normalized_return = 0.5  # 默认值
        
        # 综合评分
        composite_score = (base_score * self.base_weight + 
                          normalized_return * self.performance_weight)
        
        # 更新基础评分（用于后续使用）
        self.strategy_base_scores[strategy_id] = base_score
        
        return min(composite_score, 1.0)  # 确保不超过1
    
    def get_strategy_performance_stats(self, strategy_id: int) -> Dict[str, Any]:
        """获取策略表现统计"""
        returns = list(self.strategy_performance[strategy_id])
        if len(returns) == 0:
            return {
                'count': 0,
                'mean': 0,
                'std': 0,
                'sharpe': 0,
                'win_rate': 0
            }
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # 夏普比率（假设无风险利率为0）
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        # 胜率
        win_rate = np.sum(returns_array > 0) / len(returns_array)
        
        return {
            'count': len(returns),
            'mean': mean_return,
            'std': std_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'base_score': self.strategy_base_scores.get(strategy_id, 0.5),
            'current_score': self._calculate_enhanced_adaptation_score(strategy_id, 'SIDEWAYS')
        }
    
    def get_all_strategy_stats(self) -> Dict[int, Dict[str, Any]]:
        """获取所有策略的统计信息"""
        stats = {}
        for strategy_id in range(6):
            stats[strategy_id] = self.get_strategy_performance_stats(strategy_id)
        return stats
    
    def reset(self):
        """重置奖励计算器（每个episode开始时调用）"""
        self.prev_portfolio_value = None
        self.prev_strategy = None
        self.strategy_hold_periods = 0
        self.daily_returns = []
        self.peak_value = 0
        self.benchmark_value = None
        self.last_directional_hit_rate = 0.5
        self.last_directional_exposure_ratio = 0.0
        
        # 默认每个episode清空历史，避免跨episode目标非平稳。
        if not self.preserve_history_across_episodes:
            self.strategy_performance = {i: deque(maxlen=self.history_window) for i in range(6)}
            self.strategy_returns = {i: [] for i in range(6)}
            self.strategy_usage_count = {i: 0 for i in range(6)}
            self.recent_strategies = deque(maxlen=50)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'peak_value': self.peak_value,
            'current_drawdown': (self.peak_value - self.prev_portfolio_value) / self.peak_value 
                               if self.peak_value > 0 else 0,
            'volatility': np.std(self.daily_returns) if len(self.daily_returns) > 1 else 0,
            'strategy_hold_periods': self.strategy_hold_periods,
            'num_returns': len(self.daily_returns),
            'directional_hit_rate': float(self.last_directional_hit_rate),
            'directional_exposure_ratio': float(self.last_directional_exposure_ratio)
        }
    
    def __str__(self) -> str:
        return (f"RewardCalculator(alpha={self.alpha}, beta={self.beta}, "
                f"gamma={self.gamma}, delta={self.delta})")
    
    def calculate_transaction_costs(self, 
                                   trade_value: float, 
                                   is_buy: bool = True,
                                   volatility: float = 0.02) -> Dict[str, float]:
        """
        计算交易成本（文档要求的功能）
        
        文档要求包含：
        1. 佣金成本
        2. 滑点成本
        3. 市场冲击成本
        4. 印花税（仅卖出时）
        
        Args:
            trade_value: 交易金额
            is_buy: 是否为买入交易
            volatility: 市场波动率
        
        Returns:
            cost_details: 各项成本明细
        """
        # 1. 佣金成本
        commission = trade_value * self.commission_rate
        commission = max(commission, self.min_commission)
        
        # 2. 滑点成本（假设为交易金额的一定比例）
        slippage = trade_value * self.slippage_rate
        
        # 3. 市场冲击成本（与交易金额和波动率相关）
        market_impact = trade_value * self.market_impact_factor * (1 + volatility * 10)
        
        # 4. 印花税（仅卖出时收取）
        tax = trade_value * self.tax_rate if not is_buy else 0
        
        # 总成本
        total_cost = commission + slippage + market_impact + tax
        
        cost_details = {
            'commission': commission,
            'slippage': slippage,
            'market_impact': market_impact,
            'tax': tax,
            'total_cost': total_cost,
            'cost_rate': total_cost / trade_value if trade_value > 0 else 0
        }
        
        # 更新统计
        self.total_commission += commission
        self.total_slippage += slippage
        self.total_market_impact += market_impact
        self.total_tax += tax
        self.trade_count += 1
        
        return cost_details
    
    def evaluate_trading_costs(self, portfolio_value: float) -> Dict[str, Any]:
        """
        评估交易成本（文档要求的功能）
        
        文档要求评估：
        1. 成本占比（总成本/组合价值）
        2. 成本结构分析
        3. 成本效率评估
        4. 成本优化建议
        
        Args:
            portfolio_value: 当前组合价值
        
        Returns:
            cost_evaluation: 成本评估结果
        """
        total_cost = (self.total_commission + self.total_slippage + 
                     self.total_market_impact + self.total_tax)
        
        # 成本占比
        cost_ratio = total_cost / portfolio_value if portfolio_value > 0 else 0
        
        # 成本结构
        cost_structure = {
            'commission_ratio': self.total_commission / total_cost if total_cost > 0 else 0,
            'slippage_ratio': self.total_slippage / total_cost if total_cost > 0 else 0,
            'market_impact_ratio': self.total_market_impact / total_cost if total_cost > 0 else 0,
            'tax_ratio': self.total_tax / total_cost if total_cost > 0 else 0
        }
        
        # 成本效率评估
        avg_trade_cost = total_cost / self.trade_count if self.trade_count > 0 else 0
        
        # 成本优化建议
        optimization_suggestions = []
        if cost_ratio > 0.01:  # 成本占比超过1%
            optimization_suggestions.append("成本占比过高，建议减少交易频率")
        
        if cost_structure['slippage_ratio'] > 0.3:  # 滑点成本占比超过30%
            optimization_suggestions.append("滑点成本过高，建议优化交易时机")
        
        if cost_structure['market_impact_ratio'] > 0.2:  # 市场冲击成本占比超过20%
            optimization_suggestions.append("市场冲击成本过高，建议拆分大额订单")
        
        if self.trade_count > 0 and avg_trade_cost > 100:  # 平均单笔成本超过100元
            optimization_suggestions.append("单笔交易成本过高，建议优化交易规模")
        
        cost_evaluation = {
            'total_cost': total_cost,
            'cost_ratio': cost_ratio,
            'cost_structure': cost_structure,
            'trade_count': self.trade_count,
            'avg_trade_cost': avg_trade_cost,
            'cost_details': {
                'commission': self.total_commission,
                'slippage': self.total_slippage,
                'market_impact': self.total_market_impact,
                'tax': self.total_tax
            },
            'optimization_suggestions': optimization_suggestions,
            'cost_efficiency': self._calculate_cost_efficiency(portfolio_value, total_cost)
        }
        
        return cost_evaluation
    
    def _calculate_cost_efficiency(self, portfolio_value: float, total_cost: float) -> Dict[str, float]:
        """计算成本效率指标"""
        if self.trade_count == 0 or portfolio_value == 0:
            return {
                'cost_per_trade': 0,
                'cost_per_value': 0,
                'turnover_ratio': 0,
                'cost_effectiveness': 0
            }
        
        # 假设总交易额为组合价值的2倍（简化计算）
        total_turnover = portfolio_value * 2
        
        cost_efficiency = {
            'cost_per_trade': total_cost / self.trade_count,
            'cost_per_value': total_cost / portfolio_value,
            'turnover_ratio': total_turnover / portfolio_value,
            'cost_effectiveness': portfolio_value / total_cost if total_cost > 0 else 0
        }
        
        return cost_efficiency
    
    def get_cost_statistics(self) -> Dict[str, Any]:
        """获取交易成本统计"""
        return {
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_market_impact': self.total_market_impact,
            'total_tax': self.total_tax,
            'trade_count': self.trade_count,
            'total_cost': (self.total_commission + self.total_slippage + 
                          self.total_market_impact + self.total_tax)
        }
    
    def reset_costs(self):
        """重置交易成本统计"""
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_market_impact = 0.0
        self.total_tax = 0.0
        self.trade_count = 0
    
    def __str__(self) -> str:
        return (f"RewardCalculator(alpha={self.alpha}, beta={self.beta}, "
                f"gamma={self.gamma}, delta={self.delta})")
    
    def __repr__(self) -> str:
        return self.__str__()
