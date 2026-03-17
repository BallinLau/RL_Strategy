"""
双均线/MACD策略

结合均线交叉和MACD指标进行趋势跟踪
"""

import numpy as np
from typing import Dict, Any, List
from .base_strategy import BaseStrategy


class DualMAStrategy(BaseStrategy):
    """双均线/MACD策略（趋势策略）"""
    
    def __init__(self, fast_ma: int = 5, slow_ma: int = 20):
        """
        初始化双均线策略（增强版：支持波动率调节）
        
        Args:
            fast_ma: 快速均线周期
            slow_ma: 慢速均线周期
        """
        super().__init__("DualMA_MACD")
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        
        # 设置策略类型为趋势策略
        self.strategy_type = 'TREND'
    
    def generate_signal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成双均线+MACD信号
        
        逻辑：
        1. 当短期均线上穿长期均线 且 MACD金叉时，买入
        2. 当短期均线下穿长期均线 且 MACD死叉时，卖出
        """
        indicators = state.get('indicators', {})
        price = state.get('price', {})
        
        # 获取指标
        ma_fast = indicators.get(f'MA{self.fast_ma}', 0)
        ma_slow = indicators.get(f'MA{self.slow_ma}', 0)
        macd_cross = indicators.get('MACD_CROSS', 0)
        macd_hist = indicators.get('MACD_HIST', 0)
        macd_dif = indicators.get('MACD_DIF', 0)
        macd_dea = indicators.get('MACD_DEA', 0)
        close = price.get('close', 0)
        
        # 计算均线交叉角度（用于信号强度）
        ma_diff = (ma_fast - ma_slow) / ma_slow if ma_slow != 0 else 0
        ma_angle = np.arctan(ma_diff) * 180 / np.pi  # 转换为角度
        
        # 初始化信号
        signal_strength = 0.0
        direction = 'NEUTRAL'
        confidence = 0.0
        metadata = {
            'ma_fast': ma_fast,
            'ma_slow': ma_slow,
            'ma_diff': ma_diff,
            'ma_angle': ma_angle,
            'macd_cross': macd_cross,
            'macd_hist': macd_hist,
            'macd_dif': macd_dif,
            'macd_dea': macd_dea
        }
        
        # 买入信号：均线金叉 + MACD金叉
        if ma_fast > ma_slow and macd_cross == 1:
            signal_strength = min(abs(ma_angle) / 45, 1.0)  # 角度越大信号越强
            signal_strength += min(abs(macd_hist) / 0.5, 0.5)  # MACD柱状图越大越强
            signal_strength = min(signal_strength, 1.0)
            direction = 'LONG'
            confidence = 0.7
            metadata['reason'] = 'golden_cross'
        
        # 卖出信号：均线死叉 + MACD死叉
        elif ma_fast < ma_slow and macd_cross == -1:
            signal_strength = min(abs(ma_angle) / 45, 1.0)
            signal_strength += min(abs(macd_hist) / 0.5, 0.5)
            signal_strength = min(signal_strength, 1.0)
            direction = 'SHORT'
            confidence = 0.7
            metadata['reason'] = 'death_cross'
        
        # 持有信号：均线方向一致但无交叉
        elif ma_fast > ma_slow and macd_hist > 0:
            signal_strength = 0.5
            direction = 'LONG'
            confidence = 0.5
            metadata['reason'] = 'uptrend_hold'
        
        elif ma_fast < ma_slow and macd_hist < 0:
            signal_strength = 0.5
            direction = 'SHORT'
            confidence = 0.5
            metadata['reason'] = 'downtrend_hold'
        
        # 中性信号
        else:
            signal_strength = 0.2
            direction = 'NEUTRAL'
            confidence = 0.3
            metadata['reason'] = 'neutral'
        
        # 根据波动率状态调节信号强度（文档要求的功能）
        market_state = state.get('market_state', {})
        adjusted_strength = self.adjust_signal_by_volatility(signal_strength, market_state)
        
        # 记录波动率调节信息
        metadata['original_strength'] = signal_strength
        metadata['adjusted_strength'] = adjusted_strength
        metadata['volatility_regime'] = market_state.get('volatility_regime', 'MEDIUM')
        metadata['adjustment_factor'] = adjusted_strength / signal_strength if signal_strength > 0 else 1.0
        
        signal = {
            'signal_strength': adjusted_strength,  # 使用调节后的信号强度
            'direction': direction,
            'confidence': confidence,
            'suggested_positions': {},
            'metadata': metadata
        }
        
        self.record_signal(signal)
        return signal
    
    def get_required_indicators(self) -> List[str]:
        """返回所需指标"""
        return [
            f'MA{self.fast_ma}',
            f'MA{self.slow_ma}',
            'MACD_DIF',
            'MACD_DEA',
            'MACD_HIST',
            'MACD_CROSS'
        ]
