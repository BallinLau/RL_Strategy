"""
策略基类

定义所有交易策略的通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseStrategy(ABC):
    """交易策略基类"""
    
    def __init__(self, name: str):
        """
        初始化策略（增强版：支持波动率调节）
        
        Args:
            name: 策略名称
        """
        self.name = name
        self.signal_history = []
        
        # 波动率调节参数（文档要求的功能）
        self.volatility_adjustment = {
            'HIGH': 1.3,   # 高波动：趋势策略系数上升
            'MEDIUM': 1.0, # 中波动：不变
            'LOW': 0.7     # 低波动：反转策略系数上升
        }
        
        # 策略类型：趋势策略或反转策略
        self.strategy_type = 'TREND'  # 默认趋势策略，子类可覆盖
    
    @abstractmethod
    def generate_signal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成交易信号
        
        Args:
            state: 包含价格、指标、持仓等信息的状态字典
                {
                    'price': {'open': x, 'high': x, 'low': x, 'close': x, 'volume': x},
                    'indicators': {indicator_name: value, ...},
                    'positions': {'quantity': x, 'avg_cost': x, 'pnl': x},
                    'market_state': {'state': 'BULL', 'volatility': x, ...}
                }
        
        Returns:
            signal: 交易信号字典
                {
                    'signal_strength': float (0-1),  # 信号强度
                    'direction': str ('LONG'/'SHORT'/'NEUTRAL'),  # 方向
                    'confidence': float (0-1),  # 置信度
                    'suggested_positions': Dict[str, float],  # 建议仓位 {symbol: weight}
                    'metadata': Dict[str, Any]  # 额外信息
                }
        """
        pass
    
    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """
        返回该策略需要的技术指标列表
        
        Returns:
            indicators: 指标名称列表
        """
        pass
    
    def reset(self):
        """重置策略状态"""
        self.signal_history = []
    
    def record_signal(self, signal: Dict[str, Any]):
        """记录信号历史"""
        self.signal_history.append(signal)
    
    def get_signal_history(self) -> List[Dict[str, Any]]:
        """获取信号历史"""
        return self.signal_history
    
    def adjust_signal_by_volatility(self, signal_strength: float, 
                                   market_state: Dict[str, Any]) -> float:
        """
        根据波动率状态调节信号强度（文档要求的功能）
        
        文档要求：
        - 高波动：趋势策略系数上升（如×1.3），反转策略系数下降（如×0.7）
        - 低波动：趋势策略系数下降（如×0.7），反转策略系数上升（如×1.3）
        
        Args:
            signal_strength: 原始信号强度
            market_state: 市场状态字典，包含volatility_regime
            
        Returns:
            adjusted_strength: 调节后的信号强度
        """
        # 获取波动率状态
        vol_regime = market_state.get('volatility_regime', 'MEDIUM')
        
        # 获取调节系数
        adjustment_factor = self.volatility_adjustment.get(vol_regime, 1.0)
        
        # 如果是反转策略，调整系数相反
        if self.strategy_type == 'REVERSAL':
            if vol_regime == 'HIGH':
                adjustment_factor = 0.7  # 反转策略在高波动时系数下降
            elif vol_regime == 'LOW':
                adjustment_factor = 1.3  # 反转策略在低波动时系数上升
        
        # 应用调节
        adjusted_strength = signal_strength * adjustment_factor
        
        # 确保在0-1范围内
        return max(0, min(1, adjusted_strength))
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, type={self.strategy_type})"
    
    def __repr__(self) -> str:
        return self.__str__()
