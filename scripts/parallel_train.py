"""
并行训练脚本 - 同时训练多个时间段

将3年数据分成多个时间段，并行训练，最后合并经验
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing as mp
import yaml
import pandas as pd
from datetime import datetime, timedelta
from src.data.data_loader import DataLoader
from src.environment.trading_env import TradingEnvironment
from src.agent.ddqn_agent import DDQNAgent


def train_time_segment(args):
    """训练单个时间段"""
    segment_id, start_date, end_date, config = args
    
    print(f"[Segment {segment_id}] 训练时间段: {start_date} - {end_date}")
    
    # 加载该时间段的数据
    data_config = config.get('data', {})
    data_loader = DataLoader(data_config.get('data_path', 'data/filtered'))
    
    train_data, _, _ = data_loader.load_and_split(
        symbols=data_config.get('symbols', []),
        start_date=start_date,
        end_date=end_date,
        train_ratio=1.0,  # 全部用于训练
        val_ratio=0.0
    )
    
    # 创建环境
    env_config = config.get('environment', {})
    env = TradingEnvironment(
        data=train_data,
        initial_capital=env_config.get('initial_capital', 1000000),
        period_minutes=env_config.get('period_minutes', 240),
        switch_period_multiplier=env_config.get('switch_period_multiplier', 1),
        transaction_cost=env_config.get('transaction_cost', 0.001),
        dynamic_switch=env_config.get('dynamic_switch', True)
    )
    
    # 创建智能体
    agent_config = config.get('agent', {})
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=agent_config.get('learning_rate', 0.0001),
        gamma=agent_config.get('gamma', 0.99),
        epsilon_start=agent_config.get('epsilon_start', 1.0),
        epsilon_end=agent_config.get('epsilon_end', 0.01),
        epsilon_decay=agent_config.get('epsilon_decay', 0.995),
        target_update_freq=agent_config.get('target_update_freq', 10),
        batch_size=agent_config.get('batch_size', 64),
        buffer_capacity=agent_config.get('buffer_capacity', 100000),
        device=agent_config.get('device', 'cpu')
    )
    
    # 训练该时间段
    num_episodes = 10  # 每个时间段训练10个episodes
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) >= agent.batch_size:
                agent.train()
            
            episode_reward += reward
            state = next_state
        
        agent.decay_epsilon()
        print(f"[Segment {segment_id}] Episode {episode}: Reward={episode_reward:.2f}")
    
    # 保存该时间段的模型
    model_path = f"outputs/parallel/segment_{segment_id}_model.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)
    
    return segment_id, model_path


def merge_models(model_paths, output_path):
    """合并多个模型的经验缓冲区"""
    print("合并模型经验缓冲区...")
    
    # 加载第一个模型作为基础
    base_agent = DDQNAgent(state_dim=462, action_dim=6)  # 临时参数
    base_agent.load(model_paths[0])
    
    # 合并其他模型的经验缓冲区
    for model_path in model_paths[1:]:
        temp_agent = DDQNAgent(state_dim=462, action_dim=6)
        temp_agent.load(model_path)
        
        # 将经验添加到基础模型
        for experience in temp_agent.replay_buffer.buffer:
            base_agent.replay_buffer.buffer.append(experience)
    
    # 保存合并后的模型
    base_agent.save(output_path)
    print(f"合并完成，保存到: {output_path}")


def parallel_train():
    """并行训练主函数"""
    # 加载配置
    with open('configs/training_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 将3年数据分成6个时间段（每段6个月）
    start_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2024-12-31", "%Y-%m-%d")
    
    segments = []
    current_date = start_date
    segment_id = 1
    
    while current_date < end_date:
        segment_end = min(current_date + timedelta(days=180), end_date)  # 6个月
        segments.append((
            segment_id,
            current_date.strftime("%Y-%m-%d"),
            segment_end.strftime("%Y-%m-%d"),
            config
        ))
        current_date = segment_end
        segment_id += 1
    
    print(f"将训练分为{len(segments)}个时间段:")
    for seg_id, start, end, _ in segments:
        print(f"  Segment {seg_id}: {start} - {end}")
    
    # 并行训练
    print("\n开始并行训练...")
    with mp.Pool(processes=min(len(segments), mp.cpu_count())) as pool:
        results = pool.map(train_time_segment, segments)
    
    # 合并模型
    model_paths = [result[1] for result in results]
    merge_models(model_paths, "outputs/training/parallel_merged_model.pth")
    
    print("\n并行训练完成！")
    print("合并后的模型包含所有时间段的经验，可以直接用于继续训练。")


if __name__ == "__main__":
    parallel_train()