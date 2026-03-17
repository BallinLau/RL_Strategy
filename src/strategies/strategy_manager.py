"""
策略管理器

管理所有交易策略，提供统一的接口
"""

from typing import Dict, Any, List
from .market_neutral import MarketNeutralStrategy
from .dual_ma import DualMAStrategy
from .bollinger_bands import BollingerBandsStrategy
from .cta import CTAStrategy
from .statistical_arbitrage import StatisticalArbitrageStrategy
from .long_short_equity import LongShortEquityStrategy


class StrategyManager:
    """策略管理器"""
    
    def __init__(self):
        """初始化策略管理器"""
        self.strategies = {
            0: MarketNeutralStrategy(),
            1: DualMAStrategy(),
            2: BollingerBandsStrategy(),
            3: CTAStrategy(),
            4: StatisticalArbitrageStrategy(),
            5: LongShortEquityStrategy()
        }
        
        self.strategy_names = {
            0: "MarketNeutral",
            1: "DualMA_MACD",
            2: "BollingerBands",
            3: "CTA",
            4: "StatisticalArbitrage",
            5: "LongShortEquity"
        }
    
    def get_all_signals(self, state: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """
        获取所有策略的信号
        
        Args:
            state: 市场状态字典
        
        Returns:
            signals: {strategy_id: signal_dict}
        """
        signals = {}
        for strategy_id, strategy in self.strategies.items():
            try:
                signal = strategy.generate_signal(state)
                signals[strategy_id] = signal
            except Exception as e:
                # 如果某个策略出错，返回中性信号
                print(f"Warning: Strategy {strategy.name} failed: {e}")
                signals[strategy_id] = {
                    'signal_strength': 0.0,
                    'direction': 'NEUTRAL',
                    'confidence': 0.0,
                    'suggested_positions': {},
                    'metadata': {'error': str(e)}
                }
        
        return signals
    
    def get_strategy_by_id(self, strategy_id: int):
        """
        根据ID获取策略
        
        Args:
            strategy_id: 策略ID（0-5）
        
        Returns:
            strategy: 策略对象
        """
        return self.strategies.get(strategy_id)
    
    def get_strategy_name(self, strategy_id: int) -> str:
        """
        获取策略名称
        
        Args:
            strategy_id: 策略ID
        
        Returns:
            name: 策略名称
        """
        return self.strategy_names.get(strategy_id, "Unknown")
    
    def get_all_required_indicators(self) -> List[str]:
        """
        获取所有策略需要的指标列表
        
        Returns:
            indicators: 指标名称列表（去重）
        """
        all_indicators = set()
        for strategy in self.strategies.values():
            all_indicators.update(strategy.get_required_indicators())
        
        return sorted(list(all_indicators))
    
    def reset_all_strategies(self):
        """重置所有策略的状态"""
        for strategy in self.strategies.values():
            strategy.reset()
    
    def get_strategy_statistics(self) -> Dict[int, Dict[str, Any]]:
        """
        获取所有策略的统计信息
        
        Returns:
            stats: {strategy_id: {signal_count, avg_strength, ...}}
        """
        stats = {}
        for strategy_id, strategy in self.strategies.items():
            history = strategy.get_signal_history()
            if history:
                strengths = [s['signal_strength'] for s in history]
                confidences = [s['confidence'] for s in history]
                
                stats[strategy_id] = {
                    'name': strategy.name,
                    'signal_count': len(history),
                    'avg_strength': sum(strengths) / len(strengths),
                    'avg_confidence': sum(confidences) / len(confidences),
                    'max_strength': max(strengths),
                    'min_strength': min(strengths)
                }
            else:
                stats[strategy_id] = {
                    'name': strategy.name,
                    'signal_count': 0,
                    'avg_strength': 0.0,
                    'avg_confidence': 0.0,
                    'max_strength': 0.0,
                    'min_strength': 0.0
                }
        
        return stats
    
    def __len__(self) -> int:
        """返回策略数量"""
        return len(self.strategies)
    
    def __str__(self) -> str:
        return f"StrategyManager(strategies={len(self.strategies)})"
    
    def __repr__(self) -> str:
        return self.__str__()
