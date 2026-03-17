"""
Q网络 (Q-Network)

深度Q网络，用于估计动作价值函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class QNetwork(nn.Module):
    """
    Q网络：输入状态，输出每个动作的Q值
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.0):
        """
        初始化Q网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度（6）
            hidden_dims: 隐藏层维度列表
            dropout: Dropout比例
        """
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 构建全连接网络
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重（正常版 - 恢复正常的初始化）"""
        output_layer = None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用正常的Xavier初始化
                nn.init.xavier_uniform_(m.weight, gain=1.0)  # 恢复正常gain
                if m.bias is not None:
                    # 偏置初始化为0
                    nn.init.constant_(m.bias, 0)
                output_layer = m

        # 稳态基线偏置：轻微提高动作0先验（等权长仓基线）
        if output_layer is not None and output_layer.bias is not None and output_layer.bias.numel() > 0:
            with torch.no_grad():
                output_layer.bias[0] = 0.05
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量 (batch_size, state_dim)
        
        Returns:
            q_values: Q值张量 (batch_size, action_dim)
        """
        return self.network(state)
    
    def get_action(self, state: torch.Tensor) -> int:
        """
        获取最优动作（贪婪策略）
        
        Args:
            state: 状态张量 (state_dim,) 或 (1, state_dim)
        
        Returns:
            action: 最优动作
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()
    
    def __str__(self) -> str:
        return f"QNetwork(state_dim={self.state_dim}, action_dim={self.action_dim})"
    
    def __repr__(self) -> str:
        return self.__str__()
