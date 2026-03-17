"""
状态空间 (State Space)

将原始市场状态编码为神经网络可以处理的向量
"""

import numpy as np
from typing import Dict, Any, List
from collections import deque


class StateSpace:
    """
    状态空间编码器
    
    将包含价格、指标、持仓、市场状态等信息的字典编码为固定维度的向量
    """
    
    def __init__(self, num_stocks: int = 1, lookback_window: int = 1):
        """
        初始化状态空间
        
        Args:
            num_stocks: 股票数量
        """
        self.num_stocks = num_stocks
        self.lookback_window = max(int(lookback_window), 1)
        
        # 计算状态维度（优化后）
        # 每只股票的状态维度：
        # - 价格信息: 5维 (open, high, low, close, volume)
        # - 持仓信息: 3维 (quantity, avg_cost, pnl)
        # - 技术指标: 20维（精简后，移除冗余）
        # - 市场状态: 8维 (5维one-hot + 3维数值)
        # - 策略信号: 6维
        self.state_dim_per_stock = 42  # 从52降到42
        self.single_state_dim = self.state_dim_per_stock * num_stocks
        self.state_dim = self.single_state_dim * self.lookback_window
        self.history_buffer = deque(maxlen=self.lookback_window)
        
        # 市场状态映射
        self.market_states = ['BULL', 'BEAR', 'CORRECTION', 'REBOUND', 'SIDEWAYS']
        
        # 归一化参数（使用运行时统计）
        self.feature_mean = None
        self.feature_std = None
        self.normalization_samples = []
        self.max_normalization_samples = 1000  # 用于计算统计量的样本数
    
    def encode_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """
        将原始状态编码为向量
        
        Args:
            raw_state: 原始状态字典
                {
                    'prices': {symbol: {'open': x, 'high': x, ...}},
                    'positions': {symbol: {'quantity': x, 'avg_cost': x, 'pnl': x}},
                    'indicators': {indicator_name: value},
                    'market_state': {'state': 'BULL', 'volatility': x, ...},
                    'strategy_signals': {strategy_id: signal_strength}
                }
        
        Returns:
            state_vector: 编码后的状态向量 (state_dim,)
        """
        current_state = self._encode_single_state(raw_state)
        self.history_buffer.append(current_state)

        if len(self.history_buffer) < self.lookback_window:
            pad_count = self.lookback_window - len(self.history_buffer)
            padded = [np.zeros(self.single_state_dim, dtype=np.float32) for _ in range(pad_count)]
            seq = padded + list(self.history_buffer)
        else:
            seq = list(self.history_buffer)

        return np.concatenate(seq, axis=0).astype(np.float32)

    def _encode_single_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """编码单个时间点状态（不含时间窗口拼接）"""
        state_vector = []
        
        # 获取股票列表（如果有多只股票）
        symbols = list(raw_state.get('prices', {}).keys())
        if not symbols:
            # 如果没有价格数据，返回零向量
            return np.zeros(self.single_state_dim, dtype=np.float32)
        
        for symbol in sorted(symbols)[:self.num_stocks]:
            # 1. 价格信息 (5维)
            price_data = raw_state['prices'].get(symbol, {})
            state_vector.extend([
                price_data.get('open', 0),
                price_data.get('high', 0),
                price_data.get('low', 0),
                price_data.get('close', 0),
                price_data.get('volume', 0)
            ])
            
            # 2. 持仓信息 (3维)
            position_data = raw_state.get('positions', {}).get(symbol, {
                'quantity': 0, 'avg_cost': 0, 'pnl': 0
            })
            state_vector.extend([
                position_data.get('quantity', 0),
                position_data.get('avg_cost', 0),
                position_data.get('pnl', 0)
            ])
            
            # 3. 技术指标 (20维，精简版)
            indicators = raw_state.get('indicators_by_symbol', {}).get(
                symbol,
                raw_state.get('indicators', {})
            )
            state_vector.extend([
                # 均线（保留2个关键均线）
                indicators.get('MA10', 0),
                indicators.get('MA60', 0),
                # 布林带（保留关键指标）
                indicators.get('BB_UPPER', 0),
                indicators.get('BB_LOWER', 0),
                indicators.get('BB_PERCENT_B', 0),
                # MACD（保留完整）
                indicators.get('MACD_DIF', 0),
                indicators.get('MACD_DEA', 0),
                indicators.get('MACD_HIST', 0),
                # ADX（保留关键指标）
                indicators.get('ADX', 0),
                indicators.get('ATR', 0),
                # RSI
                indicators.get('RSI', 0),
                # 配对交易（保留关键指标）
                indicators.get('SPREAD_ZSCORE', 0),
                indicators.get('CORRELATION', 0),
                indicators.get('BETA', 0),
                # 多因子（保留关键指标）
                indicators.get('FACTOR_SCORE', 0),
                indicators.get('EXCESS_RETURN', 0),
                indicators.get('RISK_NEUTRAL_SCORE', 0),
                # 动量指标
                indicators.get('PLUS_DI', 0),
                indicators.get('MINUS_DI', 0),
                indicators.get('MACD_CROSS', 0),
            ])
            
            # 4. 市场状态 (8维: 5维one-hot + 3维数值)
            market_state = raw_state.get('market_state', {})
            state_encoding = self._encode_market_state(market_state.get('state', 'SIDEWAYS'))
            state_vector.extend(state_encoding)  # 5维
            state_vector.extend([
                market_state.get('fast_momentum', 0),
                market_state.get('slow_momentum', 0),
                market_state.get('volatility', 0)
            ])
            
            # 5. 策略信号强度 (6维)
            strategy_signals = raw_state.get('strategy_signals', {})
            for i in range(6):
                state_vector.append(strategy_signals.get(i, 0))
        
        # 如果股票数量不足，用零填充
        while len(state_vector) < self.single_state_dim:
            state_vector.append(0)

        state_array = np.array(state_vector[:self.single_state_dim], dtype=np.float32)
        
        # 应用归一化
        state_array = self._normalize_state(state_array)

        return state_array

    def reset_history(self):
        """重置时间窗口历史缓存（新episode或随机起点时调用）"""
        self.history_buffer.clear()
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        归一化状态向量（增强版 - 使用z-score标准化）
        
        Args:
            state: 原始状态向量
        
        Returns:
            normalized_state: 归一化后的状态向量
        """
        # 首先进行基础的数值清理
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 收集样本用于计算统计量
        self.normalization_samples.append(state.copy())
        if len(self.normalization_samples) > self.max_normalization_samples:
            self.normalization_samples.pop(0)
        
        # 计算运行时统计量
        if len(self.normalization_samples) >= 10:  # 至少10个样本
            samples_array = np.array(self.normalization_samples)
            self.feature_mean = np.mean(samples_array, axis=0)
            self.feature_std = np.std(samples_array, axis=0) + 1e-8  # 避免除零
        
        # 对不同类型的特征使用不同的归一化策略
        normalized_state = np.zeros_like(state)
        
        for stock_idx in range(self.num_stocks):
            start_idx = stock_idx * self.state_dim_per_stock
            
            # 1. 价格信息 (5维) - 使用相对价格 + z-score
            price_start = start_idx
            prices = state[price_start:price_start+5]
            if prices[3] > 0:  # close价格存在
                # 使用收盘价作为基准，计算相对值
                base_price = prices[3]
                relative_prices = prices[:4] / base_price
                normalized_state[price_start:price_start+4] = np.clip(relative_prices, 0.5, 1.5) - 1.0  # 转换到[-0.5, 0.5]
                
                # 成交量使用log归一化 + z-score
                if prices[4] > 0:
                    log_volume = np.log(prices[4] + 1)
                    normalized_state[price_start+4] = np.clip(log_volume / 20, 0, 1)  # 限制在[0,1]
                else:
                    normalized_state[price_start+4] = 0
            else:
                normalized_state[price_start:price_start+5] = 0
            
            # 2. 持仓信息 (3维) - 强化归一化
            pos_start = start_idx + 5
            positions = state[pos_start:pos_start+3]
            # 持仓数量：使用tanh归一化
            normalized_state[pos_start] = np.tanh(positions[0] / 1000)  # 更强的压缩
            # 平均成本：相对价格
            if prices[3] > 0 and positions[1] > 0:
                cost_ratio = positions[1] / prices[3]
                normalized_state[pos_start+1] = np.tanh(cost_ratio - 1.0)  # 以1为中心
            else:
                normalized_state[pos_start+1] = 0
            # PnL：使用tanh强压缩
            normalized_state[pos_start+2] = np.tanh(positions[2] / 10000)
            
            # 3. 技术指标 (20维) - 使用适度的z-score + 轻微压缩
            ind_start = start_idx + 8
            indicators = state[ind_start:ind_start+20]
            
            # 如果有足够的统计量，使用z-score
            if self.feature_mean is not None:
                ind_mean = self.feature_mean[ind_start:ind_start+20]
                ind_std = self.feature_std[ind_start:ind_start+20]
                z_scores = (indicators - ind_mean) / ind_std
                normalized_state[ind_start:ind_start+20] = np.clip(z_scores, -3, 3)  # 轻微裁剪，保留更多信息
            else:
                # 回退到简单裁剪
                normalized_state[ind_start:ind_start+20] = np.clip(indicators, -5, 5)
            
            # 4. 市场状态 (8维) - one-hot和数值混合
            market_start = start_idx + 28
            market_data = state[market_start:market_start+8]
            # one-hot部分保持不变
            normalized_state[market_start:market_start+5] = market_data[:5]
            # 数值部分使用tanh压缩
            normalized_state[market_start+5:market_start+8] = np.tanh(market_data[5:8])
            
            # 5. 策略信号 (6维) - 使用tanh强压缩
            signal_start = start_idx + 36
            signals = state[signal_start:signal_start+6]
            normalized_state[signal_start:signal_start+6] = np.tanh(signals * 0.5)
        
        # 最终安全检查：确保所有值都在合理范围内（放宽限制）
        normalized_state = np.clip(normalized_state, -5.0, 5.0)
        
        return normalized_state
    
    def _encode_market_state(self, state: str) -> List[float]:
        """
        将市场状态编码为one-hot向量
        
        Args:
            state: 市场状态字符串
        
        Returns:
            encoding: one-hot编码 (5维)
        """
        encoding = [0.0] * len(self.market_states)
        if state in self.market_states:
            idx = self.market_states.index(state)
            encoding[idx] = 1.0
        return encoding
    
    def get_state_dim(self) -> int:
        """获取状态维度"""
        return self.state_dim
    
    def __str__(self) -> str:
        return (
            f"StateSpace(num_stocks={self.num_stocks}, "
            f"lookback_window={self.lookback_window}, "
            f"state_dim={self.state_dim})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()
