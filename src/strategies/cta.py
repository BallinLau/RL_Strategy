"""
CTA趋势跟踪策略

基于ADX和DI指标的趋势跟踪
"""

import numpy as np
from typing import Dict, Any, List
from .base_strategy import BaseStrategy


class CTAStrategy(BaseStrategy):
    """CTA趋势跟踪策略（文档要求的功能）"""
    
    def __init__(self, adx_threshold: float = 25, atr_multiplier: float = 2.0,
                 adx_sideways_threshold: float = 20, trailing_stop_atr: float = 1.5):
        """
        初始化CTA策略（增强版）
        
        Args:
            adx_threshold: ADX趋势强度阈值（>25强趋势）
            atr_multiplier: ATR止损倍数（入场价-2×ATR）
            adx_sideways_threshold: ADX震荡阈值（<20震荡）
            trailing_stop_atr: 移动止损ATR倍数（最高点回撤1.5×ATR）
        """
        super().__init__("CTA")
        self.adx_threshold = adx_threshold
        self.atr_multiplier = atr_multiplier
        self.adx_sideways_threshold = adx_sideways_threshold
        self.trailing_stop_atr = trailing_stop_atr
        
        # 设置策略类型为趋势策略
        self.strategy_type = 'TREND'
        
        # 跟踪状态
        self.entry_price = None
        self.highest_price = None
        self.lowest_price = None
        self.position = None
    
    def generate_signal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成CTA趋势跟踪信号（文档要求的功能）
        
        文档要求：
        1. 信号强度 = ADX趋势强度 + DI方向一致性
        2. ADX > 25确认强趋势，ADX < 20减少交易
        3. 使用ATR设置止损（入场价-2×ATR）和移动止损（最高点回撤1.5×ATR）
        4. 根据波动率状态调节信号强度
        """
        indicators = state.get('indicators', {})
        price = state.get('price', {})
        market_state = state.get('market_state', {})
        
        # 获取指标
        adx = indicators.get('ADX', 0)
        plus_di = indicators.get('PLUS_DI', 0)
        minus_di = indicators.get('MINUS_DI', 0)
        atr = indicators.get('ATR', 0)
        close = price.get('close', 0)
        ma20 = indicators.get('MA20', close)
        
        # 计算ADX趋势强度成分
        adx_strength = max(0, (adx - self.adx_threshold) / (100 - self.adx_threshold))
        
        # 计算DI方向一致性成分
        if plus_di + minus_di > 0:
            di_consistency = abs(plus_di - minus_di) / (plus_di + minus_di)
        else:
            di_consistency = 0
        
        # 计算原始信号强度（文档要求：ADX趋势强度 + DI方向一致性）
        original_strength = adx_strength * 0.6 + di_consistency * 0.4
        original_strength = min(original_strength, 1.0)
        
        # 根据波动率状态调节信号强度
        adjusted_strength = self.adjust_signal_by_volatility(original_strength, market_state)
        
        # 检查止损条件
        stop_loss_hit = False
        trailing_stop_hit = False
        
        if self.position == 'LONG' and self.entry_price is not None:
            # 初始止损
            stop_loss_price = self.entry_price - self.atr_multiplier * atr
            if close <= stop_loss_price:
                stop_loss_hit = True
            
            # 移动止损
            if self.highest_price is not None:
                trailing_stop_price = self.highest_price - self.trailing_stop_atr * atr
                if close <= trailing_stop_price:
                    trailing_stop_hit = True
        
        elif self.position == 'SHORT' and self.entry_price is not None:
            # 初始止损
            stop_loss_price = self.entry_price + self.atr_multiplier * atr
            if close >= stop_loss_price:
                stop_loss_hit = True
            
            # 移动止损
            if self.lowest_price is not None:
                trailing_stop_price = self.lowest_price + self.trailing_stop_atr * atr
                if close >= trailing_stop_price:
                    trailing_stop_hit = True
        
        # 初始化信号
        signal_strength = adjusted_strength
        direction = 'NEUTRAL'
        confidence = 0.0
        metadata = {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'atr': atr,
            'close': close,
            'ma20': ma20,
            'stop_loss_atr': self.atr_multiplier * atr,
            'trailing_stop_atr': self.trailing_stop_atr,
            'adx_strength': adx_strength,
            'di_consistency': di_consistency,
            'original_strength': original_strength,
            'adjusted_strength': adjusted_strength,
            'volatility_regime': market_state.get('volatility_regime', 'MEDIUM'),
            'position': self.position,
            'entry_price': self.entry_price,
            'stop_loss_hit': stop_loss_hit,
            'trailing_stop_hit': trailing_stop_hit
        }
        
        # 止损条件
        if stop_loss_hit or trailing_stop_hit:
            signal_strength = 0.0
            direction = 'NEUTRAL'
            confidence = 0.9
            metadata['reason'] = 'stop_loss' if stop_loss_hit else 'trailing_stop'
            # 重置持仓状态
            self.position = None
            self.entry_price = None
            self.highest_price = None
            self.lowest_price = None
        
        # 强趋势条件（文档要求：ADX > 25）
        elif adx > self.adx_threshold:
            # 上升趋势（+DI > -DI）
            if plus_di > minus_di:
                # 等待回调至MA20或突破近期高点（文档要求）
                if close >= ma20:
                    direction = 'LONG'
                    confidence = 0.8
                    metadata['reason'] = 'uptrend_confirmed'
                    
                    # 更新持仓状态
                    if self.position != 'LONG':
                        self.position = 'LONG'
                        self.entry_price = close
                        self.highest_price = close
                        self.lowest_price = close
                    else:
                        self.highest_price = max(self.highest_price, close)
                else:
                    signal_strength = 0.3
                    direction = 'LONG'
                    confidence = 0.5
                    metadata['reason'] = 'uptrend_pullback'
            
            # 下降趋势（-DI > +DI）
            elif minus_di > plus_di:
                if close <= ma20:
                    direction = 'SHORT'
                    confidence = 0.8
                    metadata['reason'] = 'downtrend_confirmed'
                    
                    # 更新持仓状态
                    if self.position != 'SHORT':
                        self.position = 'SHORT'
                        self.entry_price = close
                        self.highest_price = close
                        self.lowest_price = close
                    else:
                        self.lowest_price = min(self.lowest_price, close)
                else:
                    signal_strength = 0.3
                    direction = 'SHORT'
                    confidence = 0.5
                    metadata['reason'] = 'downtrend_pullback'
        
        # 震荡市场（文档要求：ADX < 20减少交易）
        elif adx < self.adx_sideways_threshold:
            signal_strength = 0.0
            direction = 'NEUTRAL'
            confidence = 0.8
            metadata['reason'] = 'sideways_market'
        
        # 弱趋势
        else:
            signal_strength = 0.2
            direction = 'NEUTRAL'
            confidence = 0.4
            metadata['reason'] = 'weak_trend'
        
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
            'ADX',
            'PLUS_DI',
            'MINUS_DI',
            'ATR',
            'MA20'
        ]
