"""
DDQN智能体 (DDQN Agent)

Double Deep Q-Network智能体实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import Optional

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer


class DDQNAgent:
    """
    Double DQN智能体
    
    使用两个网络：
    - Q网络（在线网络）：用于选择动作
    - 目标网络：用于评估Q值
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 target_update_freq: int = 10,
                 batch_size: int = 64,
                 buffer_capacity: int = 100000,
                 device: str = None,
                 train_freq: int = 4,
                 exploration_mode: str = "prior_guided",
                 action0_bias: float = 0.2,
                 signal_blend_weight: float = 0.12,
                 random_action_temperature: float = 0.8,
                 loss_type: str = "huber",
                 gradient_clip_norm: float = 0.5,
                 valid_action_ids: Optional[list] = None):
        """
        初始化DDQN智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最小探索率
            epsilon_decay: 探索率衰减
            target_update_freq: 目标网络更新频率（episode）
            batch_size: 批次大小
            buffer_capacity: 经验池容量
            device: 计算设备
            train_freq: 训练频率（每N步训练一次）
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.train_freq = train_freq  # 新增：训练频率控制
        self.exploration_mode = str(exploration_mode)
        self.action0_bias = float(action0_bias)
        self.signal_blend_weight = float(signal_blend_weight)
        self.random_action_temperature = max(float(random_action_temperature), 1e-3)
        self.loss_type = str(loss_type).lower()
        self.gradient_clip_norm = max(float(gradient_clip_norm), 1e-3)
        if valid_action_ids is None:
            valid_action_ids = list(range(action_dim))
        valid_action_ids = sorted({
            int(a) for a in valid_action_ids
            if isinstance(a, (int, np.integer)) and 0 <= int(a) < action_dim
        })
        self.valid_action_ids = valid_action_ids if valid_action_ids else list(range(action_dim))
        self.invalid_action_ids = [a for a in range(action_dim) if a not in self.valid_action_ids]
        
        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # Q网络和目标网络
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 训练统计
        self.episode_count = 0
        self.training_step = 0
        self.total_loss = 0
        self.step_counter = 0  # 新增：步数计数器，用于控制训练频率
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（改进的epsilon-greedy + 动作平衡）
        
        Args:
            state: 当前状态
            training: 是否训练模式
        
        Returns:
            action: 选择的动作
        """
        # 关键修复：推理阶段关闭Dropout，避免评估/回测动作随机漂移
        if training:
            self.q_network.train()
        else:
            self.q_network.eval()

        state_arr = np.asarray(state, dtype=np.float32).flatten()
        # 兼容训练/测试股票池不一致导致的状态维度差异
        if state_arr.shape[0] < self.state_dim:
            pad_width = self.state_dim - state_arr.shape[0]
            state_arr = np.pad(state_arr, (0, pad_width), mode='constant', constant_values=0.0)
        elif state_arr.shape[0] > self.state_dim:
            state_arr = state_arr[:self.state_dim]

        if training and random.random() < self.epsilon:
            return self._sample_exploration_action(state_arr)
        return self._select_greedy_action(state_arr)

    def _extract_strategy_signal_prior(self, state_arr: np.ndarray) -> np.ndarray:
        """从状态向量提取策略先验（跨全部slot聚合后再加动作0偏置）。"""
        prior = np.zeros(self.action_dim, dtype=np.float32)
        slot_dim = 42
        signal_offset = 36
        signal_dim = min(6, self.action_dim)

        if state_arr.shape[0] >= slot_dim and signal_dim > 0:
            total_slots = state_arr.shape[0] // slot_dim
            signal_rows = []
            weights = []
            for slot_idx in range(total_slots):
                start = slot_idx * slot_dim + signal_offset
                end = start + signal_dim
                if end > state_arr.shape[0]:
                    break
                signal_slice = state_arr[start:end]
                if not np.all(np.isfinite(signal_slice)):
                    continue
                signal_rows.append(signal_slice)
                # 后面的slot是更近期状态，给予更高权重
                weights.append(float(slot_idx + 1))

            if signal_rows:
                signal_mat = np.vstack(signal_rows)
                weight_vec = np.asarray(weights, dtype=np.float32)
                weight_sum = float(np.sum(weight_vec))
                if weight_sum > 0:
                    aggregated = (signal_mat * weight_vec[:, None]).sum(axis=0) / weight_sum
                    prior[:signal_dim] = aggregated.astype(np.float32)

        if self.action_dim > 0:
            prior[0] += self.action0_bias
        return prior

    def _select_greedy_action(self, state_arr: np.ndarray) -> int:
        """利用阶段：Q值与策略信号先验融合，降低退化为噪声动作的概率。"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_arr).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0).detach().cpu().numpy()
        prior = self._extract_strategy_signal_prior(state_arr)
        blended_q = q_values + self.signal_blend_weight * prior
        if self.invalid_action_ids:
            blended_q = blended_q.copy()
            blended_q[self.invalid_action_ids] = -np.inf
        return int(np.argmax(blended_q))

    def _sample_exploration_action(self, state_arr: np.ndarray) -> int:
        """探索阶段：按先验概率采样，而非强制均衡采样。"""
        if self.exploration_mode != "prior_guided":
            return int(random.choice(self.valid_action_ids))

        prior = self._extract_strategy_signal_prior(state_arr)
        logits = prior[self.valid_action_ids] / self.random_action_temperature
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        denom = float(np.sum(exp_logits))
        if denom <= 0 or not np.isfinite(denom):
            return int(random.choice(self.valid_action_ids))
        probs = exp_logits / denom
        if np.any(~np.isfinite(probs)):
            return int(random.choice(self.valid_action_ids))
        picked_idx = int(np.random.choice(len(self.valid_action_ids), p=probs))
        return int(self.valid_action_ids[picked_idx])
    
    def store_transition(self, 
                        state: np.ndarray, 
                        action: int, 
                        reward: float,
                        next_state: np.ndarray, 
                        done: bool):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self) -> float:
        """
        训练一步（优化版：添加训练频率控制）
        
        Returns:
            loss: 损失值
        """
        # 训练频率控制：每train_freq步训练一次
        self.step_counter += 1
        if self.step_counter % self.train_freq != 0:
            return 0.0
        
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0

        # 训练阶段显式开启在线网络训练模式，目标网络保持评估模式
        self.q_network.train()
        self.target_network.eval()
        
        # 采样经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            # 使用在线网络选择动作
            next_q_online = self.q_network(next_states)
            if self.invalid_action_ids:
                next_q_online = next_q_online.clone()
                next_q_online[:, self.invalid_action_ids] = -1e9
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            # 使用目标网络评估Q值
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # 计算损失（高频噪声下Huber更稳健）
        if self.loss_type == "mse":
            loss = F.mse_loss(current_q_values, target_q_values)
        else:
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # 检查loss是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            print("警告: Loss为nan或inf，跳过此次更新")
            return 0.0
        
        # Loss爆炸检测和自适应学习率调整
        if hasattr(self, 'recent_losses'):
            self.recent_losses.append(loss.item())
            if len(self.recent_losses) > 100:
                self.recent_losses.pop(0)
            
            # 如果最近的Loss平均值比初始值高太多，降低学习率
            if len(self.recent_losses) >= 100:
                recent_avg = sum(self.recent_losses) / len(self.recent_losses)
                if not hasattr(self, 'initial_loss_avg'):
                    self.initial_loss_avg = recent_avg
                elif recent_avg > self.initial_loss_avg * 2:  # Loss增长超过2倍
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.9  # 降低学习率10%
                    print(f"Loss增长过快，学习率调整为: {param_group['lr']:.6f}")
                    self.initial_loss_avg = recent_avg  # 重置基准
        else:
            self.recent_losses = [loss.item()]
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（配置化）
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.gradient_clip_norm)
        self.optimizer.step()
        
        self.training_step += 1
        self.total_loss += loss.item()
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        # 将经验回放缓冲区转换为可序列化的格式
        buffer_data = list(self.replay_buffer.buffer) if len(self.replay_buffer.buffer) > 0 else []
        
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'replay_buffer': buffer_data  # 保存经验回放缓冲区
        }, path)
        print(f"模型已保存到: {path} (包含 {len(buffer_data)} 个经验)")
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.training_step = checkpoint.get('training_step', 0)
        
        # 加载经验回放缓冲区
        if 'replay_buffer' in checkpoint:
            buffer_data = checkpoint['replay_buffer']
            self.replay_buffer.buffer.clear()
            for experience in buffer_data:
                self.replay_buffer.buffer.append(experience)
            print(f"模型已从 {path} 加载 (包含 {len(buffer_data)} 个经验)")
        else:
            print(f"模型已从 {path} 加载 (经验回放缓冲区为空)")
        print(f"当前状态: episode={self.episode_count}, epsilon={self.epsilon:.3f}, buffer_size={len(self.replay_buffer)}")
    
    def get_statistics(self) -> dict:
        """获取训练统计信息"""
        avg_loss = self.total_loss / max(self.training_step, 1)
        return {
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'avg_loss': avg_loss,
            'buffer_size': len(self.replay_buffer)
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_loss = 0
    
    def __str__(self) -> str:
        return (f"DDQNAgent(state_dim={self.state_dim}, action_dim={self.action_dim}, "
                f"epsilon={self.epsilon:.3f}, episodes={self.episode_count})")
    
    def __repr__(self) -> str:
        return self.__str__()
