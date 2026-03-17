"""
策略模块

包含6种交易策略的实现：
1. 市场中性策略（配对交易）
2. 双均线/MACD策略
3. 布林带策略
4. CTA趋势跟踪策略
5. 统计套利策略
6. 股票多空策略
"""

from .base_strategy import BaseStrategy
from .market_neutral import MarketNeutralStrategy
from .dual_ma import DualMAStrategy
from .bollinger_bands import BollingerBandsStrategy
from .cta import CTAStrategy
from .statistical_arbitrage import StatisticalArbitrageStrategy
from .long_short_equity import LongShortEquityStrategy
from .strategy_manager import StrategyManager

__all__ = [
    'BaseStrategy',
    'MarketNeutralStrategy',
    'DualMAStrategy',
    'BollingerBandsStrategy',
    'CTAStrategy',
    'StatisticalArbitrageStrategy',
    'LongShortEquityStrategy',
    'StrategyManager'
]
