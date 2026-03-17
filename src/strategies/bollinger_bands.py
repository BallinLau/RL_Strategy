"""
布林带策略

基于布林带的超买超卖判断
"""

import numpy as np
from typing import Dict, Any, List
from .base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """布林带策略（文档要求的功能）"""
    
    def __init__(self, oversold_threshold: float = 0.2, 
                 overbought_threshold: float = 0.8,
                 bandwidth_narrow_threshold: float = 0.1,
                 bandwidth_wide_threshold: float = 0.3):
        """
        初始化布林带策略（增强版）
        
        Args:
            oversold_threshold: 超卖阈值（%b）
            overbought_threshold: 超买阈值（%b）
            bandwidth_narrow_threshold: 带宽收窄阈值
            bandwidth_wide_threshold: 带宽扩张阈值
        """
        super().__init__("BollingerBands")
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.bandwidth_narrow_threshold = bandwidth_narrow_threshold
        self.bandwidth_wide_threshold = bandwidth_wide_threshold
        
        # 设置策略类型为反转策略
        self.strategy_type = 'REVERSAL'
    
    def generate_signal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成布林带信号（文档要求的功能）
        
        文档要求：
        1. 信号强度 = 价格在布林带中的位置 + 带宽收窄程度
        2. 当价格触及下轨且%b < 0.2时，视为超卖，买入
        3. 当价格触及上轨且%b > 0.8时，视为超买，卖出
        4. 根据波动率状态调节信号强度
        """
        indicators = state.get('indicators', {})
        price = state.get('price', {})
        market_state = state.get('market_state', {})
        
        # 获取布林带指标
        percent_b = indicators.get('BB_PERCENT_B', 0.5)
        bb_width = indicators.get('BB_WIDTH', 0)
        bb_upper = indicators.get('BB_UPPER', 0)
        bb_middle = indicators.get('BB_MIDDLE', 0)
        bb_lower = indicators.get('BB_LOWER', 0)
        close = price.get('close', 0)
        
        # 计算价格位置成分（距离中轨的远近）
        if bb_middle > 0:
            price_position = abs(close - bb_middle) / bb_middle
        else:
            price_position = 0
        
        # 计算带宽收窄程度成分（文档要求）
        # 带宽越小（收窄）信号越强，因为可能预示着突破
        bandwidth_narrowness = max(0, (self.bandwidth_wide_threshold - bb_width) / 
                                  (self.bandwidth_wide_threshold - self.bandwidth_narrow_threshold))
        
        # 计算原始信号强度（文档要求：价格位置 + 带宽收窄程度）
        position_component = min(price_position / 0.1, 1.0)  # 假设10%的价格偏离为最大值
        narrowness_component = bandwidth_narrowness
        
        original_strength = position_component * 0.6 + narrowness_component * 0.4
        original_strength = min(original_strength, 1.0)
        
        # 根据波动率状态调节信号强度
        adjusted_strength = self.adjust_signal_by_volatility(original_strength, market_state)
        
        # 初始化信号
        signal_strength = adjusted_strength
        direction = 'NEUTRAL'
        confidence = 0.0
        metadata = {
            'percent_b': percent_b,
            'bb_width': bb_width,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'close': close,
            'price_position': price_position,
            'bandwidth_narrowness': bandwidth_narrowness,
            'original_strength': original_strength,
            'adjusted_strength': adjusted_strength,
            'volatility_regime': market_state.get('volatility_regime', 'MEDIUM')
        }
        
        # 超卖信号（文档要求：价格触及下轨且%b < 0.2）
        if percent_b < self.oversold_threshold:
            # 增强信号强度
            oversold_factor = (self.oversold_threshold - percent_b) / self.oversold_threshold
            signal_strength = min(signal_strength * (1 + oversold_factor), 1.0)
            direction = 'LONG'
            confidence = 0.7
            metadata['reason'] = 'oversold'
        
        # 超买信号（文档要求：价格触及上轨且%b > 0.8）
        elif percent_b > self.overbought_threshold:
            overbought_factor = (percent_b - self.overbought_threshold) / (1 - self.overbought_threshold)
            signal_strength = min(signal_strength * (1 + overbought_factor), 1.0)
            direction = 'SHORT'
            confidence = 0.7
            metadata['reason'] = 'overbought'
        
        # 中轨附近（中性）
        elif 0.4 < percent_b < 0.6:
            signal_strength = 0.2
            direction = 'NEUTRAL'
            confidence = 0.3
            metadata['reason'] = 'middle_band'
        
        # 其他区域（根据位置决定方向）
        else:
            if percent_b < 0.5:
                direction = 'LONG'
                metadata['reason'] = 'below_middle'
            else:
                direction = 'SHORT'
                metadata['reason'] = 'above_middle'
            confidence = 0.5
        
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
        """返回所需指标"""
        return [
            'BB_UPPER',
            'BB_MIDDLE',
            'BB_LOWER',
            'BB_WIDTH',
            'BB_PERCENT_B'
        ]
