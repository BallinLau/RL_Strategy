"""
经验回放缓冲区 (Replay Buffer)

存储和采样经验用于训练
"""

import numpy as np
import random
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        """
        初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, 
             state: np.ndarray, 
             action: int, 
             reward: float,
             next_state: np.ndarray, 
             done: bool):
        """
        添加经验
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        采样一批经验
        
        Args:
            batch_size: 批次大小
        
        Returns:
            states: 状态数组
            actions: 动作数组
            rewards: 奖励数组
            next_states: 下一状态数组
            dones: 结束标志数组
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """返回缓冲区当前大小"""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """检查是否有足够的经验进行采样"""
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    
    def __str__(self) -> str:
        return f"ReplayBuffer(size={len(self)}/{self.capacity})"
    
    def __repr__(self) -> str:
        return self.__str__()
