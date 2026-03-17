"""
统计套利策略

基于价差均值回归的套利策略
"""

import numpy as np
from typing import Dict, Any, List
from .base_strategy import BaseStrategy


class StatisticalArbitrageStrategy(BaseStrategy):
    """统计套利策略（文档要求的功能）"""
    
    def __init__(self, zscore_entry: float = 2.0, zscore_exit: float = 0.0,
                 zscore_stop: float = 3.0, max_half_life: float = 20,
                 min_half_life: float = 5.0, min_correlation: float = 0.7):
        """
        初始化统计套利策略（增强版）
        
        Args:
            zscore_entry: 开仓Z-score阈值（>2或<-2）
            zscore_exit: 平仓Z-score阈值（回归至0附近）
            zscore_stop: 止损Z-score阈值（>3或<-3）
            max_half_life: 最大半衰期（确保均值回归速度）
            min_half_life: 最小半衰期（避免过度交易）
            min_correlation: 最小相关系数要求
        """
        super().__init__("StatisticalArbitrage")
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.zscore_stop = zscore_stop
        self.max_half_life = max_half_life
        self.min_half_life = min_half_life
        self.min_correlation = min_correlation
        
        # 设置策略类型为反转策略
        self.strategy_type = 'REVERSAL'
    
    def generate_signal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成统计套利信号（文档要求的功能）
        
        文档要求：
        1. 信号强度 = Z-score绝对值 × (1 / 半衰期)
        2. 监控价差Z-score，Z-score > 2时卖出价差，Z-score < -2时买入价差
        3. 确保半衰期较短（均值回归速度快）
        4. Z-score回归至0时平仓，Z-score > 3或协整关系破裂时止损
        5. 根据波动率状态调节信号强度
        """
        indicators = state.get('indicators', {})
        market_state = state.get('market_state', {})
        
        # 获取指标
        zscore = indicators.get('SPREAD_ZSCORE', 0)
        half_life = indicators.get('HALF_LIFE', float('inf'))
        correlation = indicators.get('CORRELATION', 0)
        spread = indicators.get('SPREAD', 0)
        spread_mean = indicators.get('SPREAD_MEAN', 0)
        spread_std = indicators.get('SPREAD_STD', 1)
        adf_stat = indicators.get('ADF_STAT', 0)  # 协整性检验结果
        
        # 检查协整关系是否稳定
        cointegration_stable = correlation > self.min_correlation and adf_stat < -2.5  # ADF统计量小于-2.5表示协整
        
        # 检查半衰期是否合适（文档要求：确保均值回归速度）
        half_life_ok = self.min_half_life <= half_life <= self.max_half_life
        
        # 计算原始信号强度（文档要求：Z-score绝对值 × (1 / 半衰期)）
        zscore_component = min(abs(zscore) / self.zscore_stop, 1.0)
        
        # 半衰期成分：半衰期越短信号越强
        if half_life > 0:
            half_life_component = min((1 / half_life) * 10, 1.0)
        else:
            half_life_component = 0
        
        original_strength = zscore_component * half_life_component
        original_strength = min(original_strength, 1.0)
        
        # 根据波动率状态调节信号强度
        adjusted_strength = self.adjust_signal_by_volatility(original_strength, market_state)
        
        # 初始化信号
        signal_strength = adjusted_strength
        direction = 'NEUTRAL'
        confidence = 0.0
        metadata = {
            'zscore': zscore,
            'half_life': half_life,
            'correlation': correlation,
            'spread': spread,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'adf_stat': adf_stat,
            'cointegration_stable': cointegration_stable,
            'half_life_ok': half_life_ok,
            'zscore_component': zscore_component,
            'half_life_component': half_life_component,
            'original_strength': original_strength,
            'adjusted_strength': adjusted_strength,
            'volatility_regime': market_state.get('volatility_regime', 'MEDIUM')
        }
        
        # 止损条件（文档要求：Z-score > 3 或协整关系破裂）
        if abs(zscore) > self.zscore_stop or not cointegration_stable:
            signal_strength = 0.0
            direction = 'NEUTRAL'
            confidence = 0.9
            if abs(zscore) > self.zscore_stop:
                metadata['reason'] = 'stop_loss'
            else:
                metadata['reason'] = 'cointegration_broken'
        
        # 开仓条件（文档要求：|Z-score| > 2 且半衰期合适）
        elif abs(zscore) > self.zscore_entry and half_life_ok and cointegration_stable:
            if zscore > self.zscore_entry:
                direction = 'SELL_SPREAD'  # 卖出价差（买入弱势股，卖出强势股）
                metadata['reason'] = 'spread_too_high'
            else:
                direction = 'BUY_SPREAD'  # 买入价差（买入强势股，卖出弱势股）
                metadata['reason'] = 'spread_too_low'
            
            # 置信度基于相关系数和半衰期
            correlation_confidence = max(0, (correlation - self.min_correlation) / (1 - self.min_correlation))
            half_life_confidence = 1.0 - min(abs(half_life - 10) / 10, 1.0)  # 理想半衰期10周期
            confidence = signal_strength * (correlation_confidence * 0.6 + half_life_confidence * 0.4)
        
        # 平仓条件（文档要求：Z-score回归至0附近）
        elif abs(zscore) <= abs(self.zscore_exit):
            signal_strength = 0.0
            direction = 'NEUTRAL'
            confidence = 0.8
            metadata['reason'] = 'exit_target_reached'
        
        # 持有条件
        else:
            signal_strength = 0.3
            direction = 'HOLD'
            confidence = 0.5
            metadata['reason'] = 'hold_position'
        
        signal = {
            'signal_strength': signal_strength,
            'direction': direction,
            'confidence': confidence,
            'suggested_positions': {},
            'metadata': metadata
        }
        
        self.record_signal(signal)
        return signal
    
    def get_required_indicators(self) -> List[str]:
        """返回所需指标（文档要求的所有统计套利指标）"""
        return [
            'SPREAD',
            'SPREAD_ZSCORE',
            'SPREAD_MEAN',
            'SPREAD_STD',
            'HALF_LIFE',
            'CORRELATION',
            'ADF_STAT'  # 协整性检验结果
        ]
