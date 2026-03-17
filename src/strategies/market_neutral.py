"""
市场中性策略（配对交易）

通过配对股票的价差进行交易，保持市场中性
"""

import numpy as np
from typing import Dict, Any, List
from .base_strategy import BaseStrategy


class MarketNeutralStrategy(BaseStrategy):
    """市场中性策略（文档要求的功能）"""
    
    def __init__(self, zscore_entry: float = 2.0, zscore_exit: float = 0.5,
                 zscore_stop: float = 3.0, min_correlation: float = 0.7,
                 min_half_life: float = 10.0, max_half_life: float = 60.0):
        """
        初始化市场中性策略（增强版）
        
        Args:
            zscore_entry: 开仓Z-score阈值
            zscore_exit: 平仓Z-score阈值
            zscore_stop: 止损Z-score阈值
            min_correlation: 最小相关系数要求
            min_half_life: 最小半衰期（确保均值回归速度）
            max_half_life: 最大半衰期
        """
        super().__init__("MarketNeutral")
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.zscore_stop = zscore_stop
        self.min_correlation = min_correlation
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        
        # 设置策略类型为反转策略
        self.strategy_type = 'REVERSAL'
    
    def generate_signal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成配对交易信号（文档要求的功能）
        
        文档要求：
        1. 信号强度 = 配对价差Z-score绝对值 + 协整关系稳定性
        2. 检查半衰期确保均值回归速度
        3. 根据波动率状态调节信号强度
        """
        indicators = state.get('indicators', {})
        market_state = state.get('market_state', {})
        
        # 获取配对交易指标
        zscore = indicators.get('SPREAD_ZSCORE', 0)
        correlation = indicators.get('CORRELATION', 0)
        beta = indicators.get('BETA', 1)
        half_life = indicators.get('HALF_LIFE', float('inf'))
        adf_stat = indicators.get('ADF_STAT', 0)  # 协整性检验结果
        
        # 检查协整关系是否稳定
        cointegration_stable = correlation > self.min_correlation
        
        # 检查半衰期是否合适（文档要求：确保均值回归速度）
        half_life_ok = self.min_half_life <= half_life <= self.max_half_life
        
        # 计算信号强度（文档要求：Z-score绝对值 + 协整关系稳定性）
        zscore_component = min(abs(zscore) / self.zscore_stop, 1.0)
        cointegration_component = max(0, (correlation - self.min_correlation) / (1 - self.min_correlation))
        
        # 原始信号强度
        original_strength = zscore_component + cointegration_component
        original_strength = min(original_strength, 1.0)
        
        # 根据波动率状态调节信号强度
        adjusted_strength = self.adjust_signal_by_volatility(original_strength, market_state)
        
        # 初始化信号
        signal_strength = adjusted_strength
        direction = 'NEUTRAL'
        confidence = 0.0
        suggested_positions = {}
        metadata = {
            'zscore': zscore,
            'correlation': correlation,
            'beta': beta,
            'half_life': half_life,
            'adf_stat': adf_stat,
            'cointegration_stable': cointegration_stable,
            'half_life_ok': half_life_ok,
            'original_strength': original_strength,
            'adjusted_strength': adjusted_strength,
            'volatility_regime': market_state.get('volatility_regime', 'MEDIUM')
        }
        
        # 止损条件（文档要求：Z-score > 3 或协整关系破裂）
        if abs(zscore) > self.zscore_stop or not cointegration_stable:
            signal_strength = 0.0
            direction = 'NEUTRAL'
            confidence = 0.8
            metadata['reason'] = 'stop_loss' if abs(zscore) > self.zscore_stop else 'cointegration_broken'
        
        # 开仓条件（文档要求：|Z-score| > 2 且协整关系稳定）
        elif abs(zscore) > self.zscore_entry and cointegration_stable and half_life_ok:
            if zscore > 0:
                # 价差过大：做空股票A，买入股票B
                direction = 'SHORT_A_LONG_B'
                suggested_positions = {
                    'stock_a': -0.5,  # 做空50%
                    'stock_b': 0.5 * beta  # 买入50%*beta
                }
            else:
                # 价差过小：买入股票A，做空股票B
                direction = 'LONG_A_SHORT_B'
                suggested_positions = {
                    'stock_a': 0.5,
                    'stock_b': -0.5 * beta
                }
            
            # 置信度基于相关系数和半衰期
            correlation_confidence = (correlation - self.min_correlation) / (1 - self.min_correlation)
            half_life_confidence = 1.0 - min(abs(half_life - 30) / 30, 1.0)  # 理想半衰期30周期
            confidence = signal_strength * (correlation_confidence * 0.7 + half_life_confidence * 0.3)
            metadata['reason'] = 'entry'
        
        # 平仓条件（文档要求：|Z-score| < 0.5）
        elif abs(zscore) < self.zscore_exit:
            signal_strength = 0.0
            direction = 'NEUTRAL'
            confidence = 0.6
            metadata['reason'] = 'exit'
        
        # 持有条件
        else:
            signal_strength = 0.3
            direction = 'HOLD'
            confidence = 0.5
            metadata['reason'] = 'hold'
        
        signal = {
            'signal_strength': signal_strength,
            'direction': direction,
            'confidence': confidence,
            'suggested_positions': suggested_positions,
            'metadata': metadata
        }
        
        self.record_signal(signal)
        return signal
    
    def get_required_indicators(self) -> List[str]:
        """返回所需指标（文档要求的所有配对交易指标）"""
        return [
            'SPREAD',
            'SPREAD_ZSCORE',
            'SPREAD_MEAN',
            'SPREAD_STD',
            'CORRELATION',
            'BETA',
            'HALF_LIFE',
            'ADF_STAT',  # 协整性检验结果
            'VOLUME_RATIO'  # 成交量比
        ]
