"""
环境模块 (Environment Module)

包含强化学习交易环境的所有组件：
1. 状态空间编码
2. 动作空间定义
3. 奖励函数计算
4. 资金分配策略
5. Gym交易环境
"""

from .state_space import StateSpace
from .reward_calculator import RewardCalculator
from .position_allocator import PositionAllocator
from .trading_env import TradingEnvironment

__all__ = [
    'StateSpace',
    'RewardCalculator',
    'PositionAllocator',
    'TradingEnvironment'
]
