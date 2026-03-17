"""
训练器 (Trainer)

负责DDQN智能体的训练主循环
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime

from src.environment.trading_env import TradingEnvironment
from src.agent.ddqn_agent import DDQNAgent


class Trainer:
    """
    DDQN训练器
    
    负责：
    1. 训练主循环
    2. 日志记录
    3. 检查点保存
    4. 训练曲线绘制
    """
    
    def __init__(self,
                 env: TradingEnvironment,
                 agent: DDQNAgent,
                 config: Dict[str, Any],
                 output_dir: str = "outputs"):
        """
        初始化训练器（增强版）
        
        Args:
            env: 交易环境
            agent: DDQN智能体
            config: 训练配置
            output_dir: 输出目录
        """
        self.env = env
        self.agent = agent
        self.config = config
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.episode_portfolio_values = []
        self.episode_strategy_distributions = []
        
        # 参数调整记录（文档要求）
        self.parameter_adjustments = []
        self.parameter_history = {
            'epsilon': [],
            'learning_rate': [],
            'batch_size': [],
            'gamma': [],
            'switch_penalty': [],
            'holding_reward': []
        }
        
        # 时间戳
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 日志文件
        self.log_file = os.path.join(output_dir, "logs", f"training_{self.timestamp}.log")
        self._init_log_file()
        
        # 记录初始参数
        self._record_parameters()
    
    def _init_log_file(self):
        """初始化日志文件"""
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log - {self.timestamp}\n")
            f.write("="*80 + "\n")
            f.write(f"Environment: {self.env}\n")
            f.write(f"Agent: {self.agent}\n")
            f.write(f"Config: {self.config}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message: str, print_to_console: bool = True):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
        
        if print_to_console:
            print(log_message)
    
    def train(self, num_episodes: int = None) -> Dict[str, Any]:
        """
        训练主循环
        
        Args:
            num_episodes: 训练episode数量，如果为None则使用配置中的值
        
        Returns:
            training_stats: 训练统计信息
        """
        if num_episodes is None:
            num_episodes = self.config.get('num_episodes', 1000)
        
        save_freq = self.config.get('save_freq', 50)
        eval_freq = self.config.get('eval_freq', 10)
        log_freq = self.config.get('log_freq', 1)
        
        self.log(f"开始训练，共 {num_episodes} 个episode")
        self.log(f"环境: {self.env}")
        self.log(f"智能体: {self.agent}")
        
        best_portfolio_value = -float('inf')
        
        for episode in range(1, num_episodes + 1):
            # 重置环境
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            episode_length = 0
            strategy_counts = {}
            
            done = False
            while not done:
                # 选择动作
                action = self.agent.select_action(state, training=True)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验
                effective_action = int(info.get('effective_action', action))
                self.agent.store_transition(state, effective_action, reward, next_state, done)
                
                # 训练智能体
                if self.agent.replay_buffer.is_ready(self.agent.batch_size):
                    loss = self.agent.train()
                    episode_loss += loss
                
                # 更新状态
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # 记录策略选择
                strategy_name = info.get('strategy', 'unknown')
                strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
            
            # 衰减探索率
            self.agent.decay_epsilon()
            
            # 更新目标网络
            if episode % self.agent.target_update_freq == 0:
                self.agent.update_target_network()
            
            # 记录统计信息
            final_portfolio_value = info.get('portfolio_value', 0)
            avg_loss = episode_loss / max(episode_length, 1)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_losses.append(avg_loss)
            self.episode_portfolio_values.append(final_portfolio_value)
            self.episode_strategy_distributions.append(strategy_counts)
            
            # 日志记录
            if episode % log_freq == 0:
                strategy_dist = self._format_strategy_distribution(strategy_counts, episode_length)
                self.log(
                    f"Episode {episode:4d}/{num_episodes} | "
                    f"Reward: {episode_reward:8.2f} | "
                    f"Length: {episode_length:4d} | "
                    f"Loss: {avg_loss:6.4f} | "
                    f"Portfolio: {final_portfolio_value:12.2f} | "
                    f"Epsilon: {self.agent.epsilon:.3f} | "
                    f"Strategies: {strategy_dist}"
                )
            
            # 评估
            if episode % eval_freq == 0:
                eval_stats = self.evaluate(num_episodes=1, render=False)
                eval_portfolio = eval_stats.get('avg_portfolio_value', 0)
                
                if eval_portfolio > best_portfolio_value:
                    best_portfolio_value = eval_portfolio
                    self.save_checkpoint(episode, is_best=True)
                    self.log(f"  [BEST] 新的最佳模型，组合价值: {eval_portfolio:,.2f}")
            
            # 保存检查点
            if episode % save_freq == 0:
                self.save_checkpoint(episode)
        
        # 训练完成
        training_time = time.time() - self.start_time
        self.log(f"训练完成！总时间: {training_time:.2f}秒")
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 返回训练统计
        return self.get_training_statistics()
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            num_episodes: 评估episode数量
            render: 是否渲染环境
        
        Returns:
            eval_stats: 评估统计信息
        """
        self.log(f"开始评估，{num_episodes} 个episode")
        
        eval_rewards = []
        eval_lengths = []
        eval_portfolio_values = []
        eval_strategy_distributions = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            strategy_counts = {}
            
            done = False
            while not done:
                # 评估模式下使用确定性策略
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # 记录策略选择
                strategy_name = info.get('strategy', 'unknown')
                strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
                
                if render:
                    self.env.render()
            
            final_portfolio_value = info.get('portfolio_value', 0)
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_portfolio_values.append(final_portfolio_value)
            eval_strategy_distributions.append(strategy_counts)
        
        # 计算统计信息
        eval_stats = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_length': np.mean(eval_lengths),
            'avg_portfolio_value': np.mean(eval_portfolio_values),
            'std_portfolio_value': np.std(eval_portfolio_values),
            'strategy_distribution': self._aggregate_strategy_distributions(eval_strategy_distributions),
            'eval_episodes': num_episodes
        }
        
        self.log(f"评估结果: "
                f"平均奖励: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f}, "
                f"平均组合价值: {eval_stats['avg_portfolio_value']:,.2f} ± {eval_stats['std_portfolio_value']:,.2f}")
        
        return eval_stats
    
    def save_checkpoint(self, episode: int, is_best: bool = False):
        """
        保存检查点
        
        Args:
            episode: episode编号
            is_best: 是否是最佳模型
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        
        # 保存智能体
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode:04d}.pth")
        self.agent.save(checkpoint_path)
        
        # 保存训练器状态
        trainer_state = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_losses': self.episode_losses,
            'episode_portfolio_values': self.episode_portfolio_values,
            'episode_strategy_distributions': self.episode_strategy_distributions,
            'timestamp': self.timestamp
        }
        trainer_path = os.path.join(checkpoint_dir, f"trainer_ep{episode:04d}.pkl")
        torch.save(trainer_state, trainer_path)
        
        # 如果是最佳模型，创建符号链接
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            if os.path.exists(best_path):
                os.remove(best_path)
            os.link(checkpoint_path, best_path)
            
            best_trainer_path = os.path.join(checkpoint_dir, "best_trainer.pkl")
            if os.path.exists(best_trainer_path):
                os.remove(best_trainer_path)
            os.link(trainer_path, best_trainer_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        self.agent.load(checkpoint_path)
        self.log(f"从 {checkpoint_path} 加载检查点")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 奖励曲线
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 滑动平均奖励
        window_size = max(1, len(self.episode_rewards) // 20)
        if window_size > 1:
            smoothed_rewards = pd.Series(self.episode_rewards).rolling(window=window_size).mean()
            axes[0, 0].plot(smoothed_rewards, 'r-', linewidth=2, label=f'MA({window_size})')
            axes[0, 0].legend()
        
        # 损失曲线
        axes[0, 1].plot(self.episode_losses)
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 组合价值曲线
        axes[0, 2].plot(self.episode_portfolio_values)
        axes[0, 2].set_title('Portfolio Value')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Value')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Episode长度
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Length')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 策略分布（最后一个episode）
        if self.episode_strategy_distributions:
            last_dist = self.episode_strategy_distributions[-1]
            if last_dist:
                strategies = list(last_dist.keys())
                counts = list(last_dist.values())
                axes[1, 1].bar(strategies, counts)
                axes[1, 1].set_title('Strategy Distribution (Last Episode)')
                axes[1, 1].set_xlabel('Strategy')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 探索率衰减
        epsilons = []
        epsilon = 1.0
        for _ in range(len(self.episode_rewards)):
            epsilons.append(epsilon)
            epsilon = max(self.agent.epsilon_end, epsilon * self.agent.epsilon_decay)
        
        axes[1, 2].plot(epsilons)
        axes[1, 2].set_title('Epsilon Decay')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Epsilon')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = os.path.join(self.output_dir, f"training_curves_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"训练曲线已保存到: {plot_path}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'avg_loss': np.mean(self.episode_losses),
            'final_portfolio_value': self.episode_portfolio_values[-1] if self.episode_portfolio_values else 0,
            'best_portfolio_value': max(self.episode_portfolio_values) if self.episode_portfolio_values else 0,
            'training_time': time.time() - self.start_time,
            'timestamp': self.timestamp,
            'agent_statistics': self.agent.get_statistics()
        }
    
    def _format_strategy_distribution(self, strategy_counts: Dict[str, int], total_steps: int) -> str:
        """格式化策略分布显示"""
        if not strategy_counts:
            return "N/A"
        
        sorted_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
        formatted = []
        for strategy, count in sorted_strategies[:3]:  # 只显示前3个
            percentage = count / total_steps * 100
            formatted.append(f"{strategy}:{percentage:.1f}%")
        
        return " ".join(formatted)
    
    def _aggregate_strategy_distributions(self, distributions: List[Dict[str, int]]) -> Dict[str, float]:
        """聚合多个episode的策略分布"""
        total_counts = {}
        total_steps = 0
        
        for dist in distributions:
            for strategy, count in dist.items():
                total_counts[strategy] = total_counts.get(strategy, 0) + count
                total_steps += count
        
        if total_steps == 0:
            return {}
        
        # 转换为百分比
        return {strategy: count / total_steps * 100 
                for strategy, count in total_counts.items()}
    
    def __str__(self) -> str:
        return (f"Trainer(env={self.env}, agent={self.agent}, "
                f"episodes={len(self.episode_rewards)})")
    
    def _record_parameters(self):
        """记录当前参数"""
        self.parameter_history['epsilon'].append(self.agent.epsilon)
        self.parameter_history['learning_rate'].append(self.agent.learning_rate)
        self.parameter_history['batch_size'].append(self.agent.batch_size)
        self.parameter_history['gamma'].append(self.agent.gamma)
        
        # 从环境中获取交易相关参数
        if hasattr(self.env, 'switch_penalty'):
            self.parameter_history['switch_penalty'].append(self.env.switch_penalty)
        else:
            self.parameter_history['switch_penalty'].append(0.0)
            
        if hasattr(self.env, 'holding_reward'):
            self.parameter_history['holding_reward'].append(self.env.holding_reward)
        else:
            self.parameter_history['holding_reward'].append(0.0)
    
    def adjust_parameters(self, episode: int, performance_metrics: Dict[str, Any]):
        """
        动态调整参数（文档要求的功能）
        
        文档要求根据性能指标动态调整：
        1. 学习率：损失波动大时降低，收敛缓慢时提高
        2. 探索率：收益不稳定时提高，稳定时降低
        3. 切换惩罚：切换频繁时增加，切换过少时减少
        4. 持仓奖励：持仓时间短时增加，持仓时间长时减少
        """
        adjustments = {}
        
        # 规则1：学习率调整
        if len(self.episode_losses) >= 10:
            recent_losses = self.episode_losses[-10:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            
            if loss_std > 0.1:  # 损失波动大
                new_lr = self.agent.learning_rate * 0.9
                adjustments['learning_rate'] = {
                    'old': self.agent.learning_rate,
                    'new': new_lr,
                    'reason': f'损失波动大 (std={loss_std:.3f})'
                }
                self.agent.learning_rate = new_lr
            elif loss_mean > 0.5 and loss_std < 0.05:  # 收敛缓慢
                new_lr = self.agent.learning_rate * 1.1
                adjustments['learning_rate'] = {
                    'old': self.agent.learning_rate,
                    'new': new_lr,
                    'reason': f'收敛缓慢 (mean={loss_mean:.3f})'
                }
                self.agent.learning_rate = new_lr
        
        # 规则2：探索率调整
        if len(self.episode_rewards) >= 20:
            recent_rewards = self.episode_rewards[-20:]
            reward_std = np.std(recent_rewards)
            
            if reward_std > np.mean(recent_rewards) * 0.5:  # 收益不稳定
                # 临时提高探索率
                old_epsilon = self.agent.epsilon
                new_epsilon = min(0.5, old_epsilon * 1.2)
                adjustments['epsilon'] = {
                    'old': old_epsilon,
                    'new': new_epsilon,
                    'reason': f'收益不稳定 (std={reward_std:.2f})'
                }
                self.agent.epsilon = new_epsilon
            elif reward_std < np.mean(recent_rewards) * 0.1:  # 收益稳定
                # 加速衰减探索率
                old_decay = self.agent.epsilon_decay
                new_decay = old_decay * 0.95  # 衰减更快
                adjustments['epsilon_decay'] = {
                    'old': old_decay,
                    'new': new_decay,
                    'reason': f'收益稳定 (std={reward_std:.2f})'
                }
                self.agent.epsilon_decay = new_decay
        
        # 规则3：切换惩罚调整
        if len(self.episode_strategy_distributions) >= 5:
            recent_dists = self.episode_strategy_distributions[-5:]
            avg_switches = self._calculate_avg_switches(recent_dists)
            
            if hasattr(self.env, 'switch_penalty'):
                if avg_switches > 15:  # 切换频繁
                    old_penalty = self.env.switch_penalty
                    new_penalty = old_penalty * 1.5
                    adjustments['switch_penalty'] = {
                        'old': old_penalty,
                        'new': new_penalty,
                        'reason': f'切换频繁 ({avg_switches:.1f}次/天)'
                    }
                    self.env.switch_penalty = new_penalty
                elif avg_switches < 5:  # 切换过少
                    old_penalty = self.env.switch_penalty
                    new_penalty = max(0.01, old_penalty * 0.7)
                    adjustments['switch_penalty'] = {
                        'old': old_penalty,
                        'new': new_penalty,
                        'reason': f'切换过少 ({avg_switches:.1f}次/天)'
                    }
                    self.env.switch_penalty = new_penalty
        
        # 规则4：持仓奖励调整
        if len(self.episode_lengths) >= 10:
            avg_holding_period = np.mean(self.episode_lengths[-10:])
            
            if hasattr(self.env, 'holding_reward'):
                if avg_holding_period < 20:  # 持仓时间短
                    old_reward = self.env.holding_reward
                    new_reward = old_reward * 1.3
                    adjustments['holding_reward'] = {
                        'old': old_reward,
                        'new': new_reward,
                        'reason': f'持仓时间短 ({avg_holding_period:.1f}步)'
                    }
                    self.env.holding_reward = new_reward
                elif avg_holding_period > 100:  # 持仓时间长
                    old_reward = self.env.holding_reward
                    new_reward = old_reward * 0.8
                    adjustments['holding_reward'] = {
                        'old': old_reward,
                        'new': new_reward,
                        'reason': f'持仓时间长 ({avg_holding_period:.1f}步)'
                    }
                    self.env.holding_reward = new_reward
        
        # 记录调整
        if adjustments:
            self.parameter_adjustments.append({
                'episode': episode,
                'timestamp': datetime.now().isoformat(),
                'adjustments': adjustments
            })
            
            # 记录参数变化
            self._record_parameters()
            
            # 记录日志
            for param, info in adjustments.items():
                self.log(f"  参数调整: {param}: {info['old']:.4f} → {info['new']:.4f} ({info['reason']})")
        
        return adjustments
    
    def _calculate_avg_switches(self, strategy_distributions: List[Dict[str, int]]) -> float:
        """计算平均切换次数"""
        total_switches = 0
        
        for dist in strategy_distributions:
            if dist:
                # 策略数量减1近似为切换次数
                total_switches += max(0, len(dist) - 1)
        
        return total_switches / len(strategy_distributions) if strategy_distributions else 0
    
    def get_parameter_history(self) -> Dict[str, List[float]]:
        """获取参数历史"""
        return self.parameter_history
    
    def get_parameter_adjustments(self) -> List[Dict[str, Any]]:
        """获取参数调整记录"""
        return self.parameter_adjustments
    
    def plot_parameter_history(self):
        """绘制参数历史曲线"""
        if not any(self.parameter_history.values()):
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 绘制每个参数的历史
        params = list(self.parameter_history.keys())
        for idx, param in enumerate(params):
            if self.parameter_history[param]:
                ax = axes[idx // 3, idx % 3]
                ax.plot(self.parameter_history[param])
                ax.set_title(f'{param} History')
                ax.set_xlabel('Adjustment Step')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                
                # 标记调整点
                for adj_idx, adjustment in enumerate(self.parameter_adjustments):
                    if param in adjustment.get('adjustments', {}):
                        ax.axvline(x=adj_idx, color='r', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = os.path.join(self.output_dir, f"parameter_history_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"参数历史曲线已保存到: {plot_path}")
    
    def __str__(self) -> str:
        return (f"Trainer(env={self.env}, agent={self.agent}, "
                f"episodes={len(self.episode_rewards)})")
    
    def __repr__(self) -> str:
        return self.__str__()
