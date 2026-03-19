"""
训练脚本 (Training Script)

完整的DDQN训练流程示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.environment.trading_env import TradingEnvironment
from src.agent.ddqn_agent import DDQNAgent
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.training.backtest import BacktestEngine


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def resolve_symbols(data_config: dict) -> list:
    """
    从配置中解析股票列表。

    优先级：
    1) symbols_file（CSV文件，优先使用）
    2) symbols（配置内显式列表）
    """
    symbols_file = data_config.get('symbols_file')
    if symbols_file:
        try:
            symbol_df = pd.read_csv(symbols_file)
            if symbol_df.empty:
                raise ValueError("symbols_file为空")

            # 常见股票代码列名，若都不存在则使用第一列
            col_candidates = ['Stkcd', 'stkcd', 'symbol', 'code', 'stock_code', 'ts_code']
            code_col = None
            for c in col_candidates:
                if c in symbol_df.columns:
                    code_col = c
                    break
            if code_col is None:
                code_col = symbol_df.columns[0]

            raw_symbols = (
                symbol_df[code_col]
                .astype(str)
                .str.strip()
                .str.replace('.0', '', regex=False)
                .tolist()
            )
            # 去重且保留顺序
            seen = set()
            symbols = []
            for s in raw_symbols:
                if not s or s.lower() == 'nan':
                    continue
                if s not in seen:
                    seen.add(s)
                    symbols.append(s)

            print(f"从symbols_file加载股票池: {symbols_file}")
            print(f"  代码列: {code_col} | 股票数: {len(symbols)}")
            return symbols
        except Exception as e:
            print(f"警告: 读取symbols_file失败 ({symbols_file}): {e}")
            print("将回退到配置中的symbols字段")

    cfg_symbols = data_config.get('symbols', [])
    return [str(s).strip().replace('.0', '') for s in cfg_symbols if str(s).strip()]


def save_episode_parameters(episode_data: list, output_dir: str):
    """
    保存每个episode的参数到JSON文件
    
    Args:
        episode_data: episode参数列表
        output_dir: 输出目录
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        json_file = os.path.join(output_dir, 'episode_parameters.json')
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)
        
        print(f"Episode参数已保存到: {json_file}")
        
    except Exception as e:
        print(f"保存episode参数时出错: {e}")
        print("跳过参数保存，继续执行...")


def plot_training_metrics(episode_data: list, output_dir: str):
    """
    绘制训练指标图表
    
    Args:
        episode_data: episode参数列表
        output_dir: 输出目录
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查数据是否为空
        if not episode_data:
            print("警告: 没有episode数据，跳过图表生成")
            return
        
        # 安全地提取数据，使用默认值防止KeyError
        episodes = [d.get('episode', i+1) for i, d in enumerate(episode_data)]
        losses = [d.get('loss', 0.0) for d in episode_data]
        total_returns = [d.get('total_return', 0.0) for d in episode_data]
        rewards = [d.get('reward', 0.0) for d in episode_data]
        epsilons = [d.get('epsilon', 1.0) for d in episode_data]
        
        # 创建4个子图
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        
        # 1. Training Loss (对数刻度)
        axes[0].plot(episodes, losses, 'b-', linewidth=1.5)
        axes[0].set_xlabel('Episode', fontsize=12)
        axes[0].set_ylabel('Loss (log scale)', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Total Return
        axes[1].plot(episodes, total_returns, 'g-', linewidth=1.5)
        axes[1].set_xlabel('Episode', fontsize=12)
        axes[1].set_ylabel('Total Return', fontsize=12)
        axes[1].set_title('Total Return', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 3. Epsilon Decay
        axes[2].plot(episodes, epsilons, 'orange', linewidth=1.5)
        axes[2].set_xlabel('Episode', fontsize=12)
        axes[2].set_ylabel('Epsilon', fontsize=12)
        axes[2].set_title('Epsilon Decay (Exploration Rate)', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Reward per Episode
        axes[3].plot(episodes, rewards, 'red', linewidth=1.5)
        axes[3].set_xlabel('Episode', fontsize=12)
        axes[3].set_ylabel('Total Reward', fontsize=12)
        axes[3].set_title('Reward per Episode', fontsize=14, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = os.path.join(output_dir, 'training_metrics.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"训练指标图表已保存到: {plot_file}")
        
    except Exception as e:
        print(f"生成训练图表时出错: {e}")
        print("跳过图表生成，继续执行...")
        # 确保matplotlib图形被关闭
        try:
            plt.close('all')
        except:
            pass


def prepare_data(config: dict) -> dict:
    """准备训练数据"""
    print("="*80)
    print("准备训练数据")
    print("="*80)
    
    # 加载数据配置
    data_config = config.get('data', {})
    data_path = data_config.get('data_path', 'data/filtered')  # 恢复原始路径
    symbols = resolve_symbols(data_config)
    train_start_date = data_config.get('train_start_date', '2022-01-01')
    train_end_date = data_config.get('train_end_date', '2023-12-31')
    train_ratio = data_config.get('train_ratio', 0.7)
    val_ratio = data_config.get('val_ratio', 0.15)
    split_mode = str(data_config.get('split_mode', 'ratio')).lower()
    
    # 训练数据聚合频率与环境保持一致，避免口径错配
    env_config = config.get('environment', {})
    period_minutes = env_config.get('period_minutes', 20)
    bar_frequency = str(data_config.get('bar_frequency', 'minute')).lower()

    # 创建数据加载器和预处理器
    data_loader = DataLoader(data_path, symbols)
    data_preprocessor = DataPreprocessor(
        period_minutes=period_minutes,
        bar_frequency=bar_frequency
    )
    
    # 加载原始数据
    print(f"加载数据: {train_start_date} 到 {train_end_date}")
    raw_data = data_loader.load_data(train_start_date, train_end_date)
    
    if raw_data.empty:
        raise ValueError(f"在 {data_path} 中未找到 {train_start_date} 到 {train_end_date} 的数据")
    
    print(f"原始数据形状: {raw_data.shape}")
    
    # 数据预处理
    if bar_frequency == 'daily':
        print("数据预处理... (聚合频率: 日度)")
    else:
        print(f"数据预处理... (聚合周期: {period_minutes}分钟)")
    processed_data = data_preprocessor.preprocess_pipeline(
        raw_data,
        handle_missing=True,
        aggregate=True,
        remove_outliers=True,
        add_time_features=False,
        calculate_returns=True
    )
    
    print(f"处理后数据形状: {processed_data.shape}")
    
    # 划分数据集
    print("划分数据集...")
    if split_mode in ('calendar_year', 'year'):
        test_year = int(data_config.get('test_year', 2025))
        fallback_to_latest = bool(data_config.get('fallback_to_latest_available_test_year', True))
        timestamp_series = pd.to_datetime(processed_data['timestamp'], errors='coerce')
        year_series = timestamp_series.dt.year

        effective_test_year = test_year
        test_mask = year_series == effective_test_year
        if not bool(test_mask.any()):
            if fallback_to_latest:
                available_years = sorted(set(int(y) for y in year_series.dropna().unique()))
                if not available_years:
                    raise ValueError("数据中没有可用年份，无法按年份划分")
                older_years = [y for y in available_years if y < test_year]
                effective_test_year = older_years[-1] if older_years else available_years[-1]
                print(f"警告: 请求测试年份 {test_year} 无数据，回退到最近可用年份 {effective_test_year}")
                test_mask = year_series == effective_test_year
            else:
                raise ValueError(f"请求测试年份 {test_year} 无数据，且未开启回退")

        pre_test_mask = year_series < effective_test_year
        if not bool(pre_test_mask.any()):
            raise ValueError(f"测试年份 {effective_test_year} 之前没有训练数据")

        pre_test_data = processed_data[pre_test_mask].sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        test_data = processed_data[test_mask].sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        unique_train_ts = pre_test_data['timestamp'].drop_duplicates().sort_values().to_numpy()
        if len(unique_train_ts) <= 1 or float(val_ratio) <= 0:
            train_data = pre_test_data
            val_data = pre_test_data.iloc[0:0].copy()
        else:
            split_idx = int(len(unique_train_ts) * (1.0 - float(val_ratio)))
            split_idx = max(1, min(split_idx, len(unique_train_ts) - 1))
            train_ts = unique_train_ts[:split_idx]
            val_ts = unique_train_ts[split_idx:]
            train_data = pre_test_data[pre_test_data['timestamp'].isin(train_ts)].reset_index(drop=True)
            val_data = pre_test_data[pre_test_data['timestamp'].isin(val_ts)].reset_index(drop=True)

        print(f"年份切分: 训练/验证 < {effective_test_year}, 测试 = {effective_test_year}")
    else:
        train_data, val_data, test_data = data_loader.split_data(
            processed_data,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )

    # 统一三个数据集的股票池，避免训练/测试状态维度不一致
    symbol_sets = []
    if not train_data.empty:
        symbol_sets.append(set(train_data['symbol'].astype(str).unique()))
    if not val_data.empty:
        symbol_sets.append(set(val_data['symbol'].astype(str).unique()))
    if not test_data.empty:
        symbol_sets.append(set(test_data['symbol'].astype(str).unique()))

    common_symbols = sorted(set.intersection(*symbol_sets)) if symbol_sets else []
    if common_symbols:
        train_data = train_data[train_data['symbol'].astype(str).isin(common_symbols)].copy()
        val_data = val_data[val_data['symbol'].astype(str).isin(common_symbols)].copy()
        test_data = test_data[test_data['symbol'].astype(str).isin(common_symbols)].copy()
        print(f"统一股票池后股票数: {len(common_symbols)}")
    else:
        print("警告: 训练/验证/测试无公共股票池，将使用原始划分结果")
    
    print(f"训练集: {train_data.shape}, 验证集: {val_data.shape}, 测试集: {test_data.shape}")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'full': processed_data
    }


def create_environment(data: pd.DataFrame, config: dict) -> TradingEnvironment:
    """创建交易环境"""
    print("\n" + "="*80)
    print("创建交易环境")
    print("="*80)
    
    env_config = config.get('environment', {})
    initial_capital = env_config.get('initial_capital', 1000000)
    period_minutes = env_config.get('period_minutes', 20)
    state_lookback_days = env_config.get('state_lookback_days', 5)
    switch_period_multiplier = env_config.get('switch_period_multiplier', 1)
    transaction_cost = env_config.get('transaction_cost', 0.001)
    dynamic_switch = env_config.get('dynamic_switch', True)
    state_update_frequency = env_config.get('state_update_frequency', 'dynamic')
    allow_short = env_config.get('allow_short', True)
    max_short_per_symbol_ratio = env_config.get('max_short_per_symbol_ratio', 0.1)
    max_total_short_ratio = env_config.get('max_total_short_ratio', 0.25)
    max_position_ratio = env_config.get('max_position_ratio', 0.3)
    total_position_ratio = env_config.get('total_position_ratio', 0.8)
    rebalance_tolerance = env_config.get('rebalance_tolerance', 0.02)
    max_trade_fraction = env_config.get('max_trade_fraction', 0.01)
    slippage_rate = env_config.get('slippage_rate', 0.0002)
    impact_rate = env_config.get('impact_rate', 0.0001)
    min_commission = env_config.get('min_commission', 2.0)
    min_signal_threshold = env_config.get('min_signal_threshold', 0.08)
    signal_power = env_config.get('signal_power', 1.5)
    profit_lock_enabled = env_config.get('profit_lock_enabled', False)
    profit_lock_min_return = env_config.get('profit_lock_min_return', 0.03)
    profit_lock_drawdown_threshold = env_config.get('profit_lock_drawdown_threshold', 0.08)
    profit_lock_cooldown_steps = env_config.get('profit_lock_cooldown_steps', 10)
    profit_lock_safe_action = env_config.get('profit_lock_safe_action', 0)
    min_action_hold_steps = env_config.get('min_action_hold_steps', 1)
    
    # 获取奖励配置
    reward_config = config.get('reward', {})
    
    env = TradingEnvironment(
        data=data,
        initial_capital=initial_capital,
        period_minutes=period_minutes,
        state_lookback_days=state_lookback_days,
        switch_period_multiplier=switch_period_multiplier,
        transaction_cost=transaction_cost,
        dynamic_switch=dynamic_switch,
        state_update_frequency=state_update_frequency,
        allow_short=allow_short,
        max_short_per_symbol_ratio=max_short_per_symbol_ratio,
        max_total_short_ratio=max_total_short_ratio,
        max_position_ratio=max_position_ratio,
        total_position_ratio=total_position_ratio,
        rebalance_tolerance=rebalance_tolerance,
        max_trade_fraction=max_trade_fraction,
        slippage_rate=slippage_rate,
        impact_rate=impact_rate,
        min_commission=min_commission,
        min_signal_threshold=min_signal_threshold,
        signal_power=signal_power,
        profit_lock_enabled=profit_lock_enabled,
        profit_lock_min_return=profit_lock_min_return,
        profit_lock_drawdown_threshold=profit_lock_drawdown_threshold,
        profit_lock_cooldown_steps=profit_lock_cooldown_steps,
        profit_lock_safe_action=profit_lock_safe_action,
        min_action_hold_steps=min_action_hold_steps,
        reward_config=reward_config
    )
    
    print(f"环境创建成功:")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  初始资金: {initial_capital:,.2f}")
    
    return env


def create_agent(env: TradingEnvironment, config: dict, resume_from: str = None) -> DDQNAgent:
    """创建DDQN智能体"""
    print("\n" + "="*80)
    print("创建DDQN智能体")
    print("="*80)
    
    agent_config = config.get('agent', {})
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    requested_device = agent_config.get('device', None)
    if isinstance(requested_device, str) and requested_device.startswith('cuda') and not torch.cuda.is_available():
        print("警告: 配置请求CUDA但当前不可用，自动回退到CPU")
        requested_device = 'cpu'

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
        device=requested_device,
        train_freq=agent_config.get('train_freq', 1),  # 添加训练频率参数，默认每步都训练
        exploration_mode=agent_config.get('exploration_mode', 'prior_guided'),
        action0_bias=agent_config.get('action0_bias', 0.2),
        signal_blend_weight=agent_config.get('signal_blend_weight', 0.12),
        random_action_temperature=agent_config.get('random_action_temperature', 0.8),
        loss_type=agent_config.get('loss_type', 'huber'),
        gradient_clip_norm=agent_config.get('gradient_clip_norm', 0.5),
        valid_action_ids=[0, 1, 2, 3, 4] if not bool(getattr(env, 'allow_short', True)) else None
    )
    
    # 如果指定了恢复路径，加载模型
    if resume_from:
        print(f"\n从检查点恢复训练: {resume_from}")
        agent.load(resume_from)
        if agent_config.get('clear_replay_buffer_on_resume', False):
            agent.replay_buffer.buffer.clear()
            print("已清空历史经验回放缓冲区（按配置）")
        if agent_config.get('reset_epsilon_on_resume', False):
            agent.epsilon = float(agent_config.get('epsilon_start', agent.epsilon))
            print(f"已重置epsilon到: {agent.epsilon:.3f}")
        print(f"已加载模型，当前episode: {agent.episode_count}, epsilon: {agent.epsilon:.3f}")
    
    print(f"智能体创建成功:")
    print(f"  状态维度: {state_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  学习率: {agent_config.get('learning_rate', 0.0001)}")
    print(f"  批次大小: {agent_config.get('batch_size', 64)}")
    print(f"  设备: {agent.device}")
    print(f"  有效动作: {agent.valid_action_ids}")
    
    return agent


def save_incremental_checkpoint(agent: DDQNAgent, episode: int, step: int, 
                               episode_reward: float, episode_loss: float, 
                               step_count: int, output_dir: str):
    """
    保存增量checkpoint
    
    Args:
        agent: DDQN智能体
        episode: 当前episode编号
        step: 当前步数
        episode_reward: 当前episode累积奖励
        episode_loss: 当前episode累积loss
        step_count: 有效训练步数
        output_dir: 输出目录
    """
    checkpoint_dir = os.path.join(output_dir, 'incremental_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_ep{episode}_step{step}.pth')
    
    # 保存完整的训练状态
    checkpoint = {
        'episode': episode,
        'step': step,
        'episode_reward': episode_reward,
        'episode_loss': episode_loss,
        'step_count': step_count,
        'q_network_state': agent.q_network.state_dict(),
        'target_network_state': agent.target_network.state_dict(),
        'optimizer_state': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'episode_count': agent.episode_count
    }
    
    torch.save(checkpoint, checkpoint_file)
    print(f"  增量checkpoint已保存: {checkpoint_file}")
    
    # 只保留最近3个checkpoint以节省空间
    cleanup_old_checkpoints(checkpoint_dir, keep_last=3)


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int = 3):
    """
    清理旧的checkpoint文件，只保留最近的N个
    
    Args:
        checkpoint_dir: checkpoint目录
        keep_last: 保留最近的N个文件
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # 获取所有checkpoint文件
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('checkpoint_ep') and f.endswith('.pth'):
            full_path = os.path.join(checkpoint_dir, f)
            checkpoint_files.append((full_path, os.path.getmtime(full_path)))
    
    # 按修改时间排序
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    # 删除旧文件
    for file_path, _ in checkpoint_files[keep_last:]:
        try:
            os.remove(file_path)
            print(f"  已删除旧checkpoint: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  删除checkpoint失败: {e}")


def load_incremental_checkpoint(agent: DDQNAgent, checkpoint_file: str) -> dict:
    """
    从增量checkpoint恢复训练状态
    
    Args:
        agent: DDQN智能体
        checkpoint_file: checkpoint文件路径
    
    Returns:
        恢复的训练状态字典
    """
    print(f"\n从增量checkpoint恢复: {checkpoint_file}")
    
    checkpoint = torch.load(checkpoint_file, map_location=agent.device)
    
    # 恢复网络权重
    agent.q_network.load_state_dict(checkpoint['q_network_state'])
    agent.target_network.load_state_dict(checkpoint['target_network_state'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    # 恢复训练状态
    agent.epsilon = checkpoint['epsilon']
    agent.episode_count = checkpoint['episode_count']
    
    resume_state = {
        'episode': checkpoint['episode'],
        'step': checkpoint['step'],
        'episode_reward': checkpoint['episode_reward'],
        'episode_loss': checkpoint['episode_loss'],
        'step_count': checkpoint['step_count']
    }
    
    print(f"已恢复到 Episode {resume_state['episode']}, Step {resume_state['step']}")
    print(f"  累积奖励: {resume_state['episode_reward']:.2f}")
    print(f"  Epsilon: {agent.epsilon:.3f}")
    
    return resume_state


def find_latest_incremental_checkpoint(output_dir: str) -> str:
    """
    查找最新的增量checkpoint
    
    Args:
        output_dir: 输出目录
    
    Returns:
        最新checkpoint的路径，如果没有则返回None
    """
    checkpoint_dir = os.path.join(output_dir, 'incremental_checkpoints')
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('checkpoint_ep') and f.endswith('.pth'):
            full_path = os.path.join(checkpoint_dir, f)
            checkpoint_files.append((full_path, os.path.getmtime(full_path)))
    
    if not checkpoint_files:
        return None
    
    # 返回最新的文件
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    return checkpoint_files[0][0]


def _build_env_config_from_config(config: dict, initial_capital: float, period_minutes: int, bar_frequency: str) -> dict:
    """统一构建环境配置，避免训练/验证/回测配置漂移。"""
    env_config = config.get('environment', {})
    return {
        'initial_capital': initial_capital,
        'period_minutes': period_minutes,
        'bar_frequency': bar_frequency,
        'state_lookback_days': env_config.get('state_lookback_days', 5),
        'switch_period_multiplier': env_config.get('switch_period_multiplier', 1),
        'transaction_cost': env_config.get('transaction_cost', 0.001),
        'dynamic_switch': env_config.get('dynamic_switch', True),
        'state_update_frequency': env_config.get('state_update_frequency', 'dynamic'),
        'allow_short': env_config.get('allow_short', True),
        'max_short_per_symbol_ratio': env_config.get('max_short_per_symbol_ratio', 0.1),
        'max_total_short_ratio': env_config.get('max_total_short_ratio', 0.25),
        'max_position_ratio': env_config.get('max_position_ratio', 0.3),
        'total_position_ratio': env_config.get('total_position_ratio', 0.8),
        'rebalance_tolerance': env_config.get('rebalance_tolerance', 0.02),
        'max_trade_fraction': env_config.get('max_trade_fraction', 0.01),
        'slippage_rate': env_config.get('slippage_rate', 0.0002),
        'impact_rate': env_config.get('impact_rate', 0.0001),
        'min_commission': env_config.get('min_commission', 2.0),
        'min_signal_threshold': env_config.get('min_signal_threshold', 0.08),
        'signal_power': env_config.get('signal_power', 1.5),
        'profit_lock_enabled': env_config.get('profit_lock_enabled', False),
        'profit_lock_min_return': env_config.get('profit_lock_min_return', 0.03),
        'profit_lock_drawdown_threshold': env_config.get('profit_lock_drawdown_threshold', 0.08),
        'profit_lock_cooldown_steps': env_config.get('profit_lock_cooldown_steps', 10),
        'profit_lock_safe_action': env_config.get('profit_lock_safe_action', 0),
        'min_action_hold_steps': env_config.get('min_action_hold_steps', 1)
    }


def _slice_recent_timestamps(data: pd.DataFrame, max_timestamps: int) -> pd.DataFrame:
    """仅保留最近N个唯一时间点，用于快速选模/教师筛选。"""
    if data is None or data.empty:
        return data
    if max_timestamps is None or int(max_timestamps) <= 0:
        return data
    max_timestamps = int(max_timestamps)

    tmp = data.copy()
    ts = pd.to_datetime(tmp['timestamp'], errors='coerce')
    tmp['_ts_slice'] = ts
    unique_ts = pd.DatetimeIndex(pd.unique(tmp['_ts_slice'].dropna())).sort_values()
    if len(unique_ts) <= max_timestamps:
        return data

    keep_ts = set(unique_ts[-max_timestamps:])
    sliced = tmp[tmp['_ts_slice'].isin(keep_ts)].drop(columns=['_ts_slice'])
    sliced = sliced.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    return sliced


def _split_oos_windows(data: pd.DataFrame, window_count: int) -> list:
    """按时间顺序把数据切成若干OOS窗口。"""
    if data is None or data.empty:
        return []
    window_count = max(int(window_count), 1)
    if window_count == 1:
        return [data]

    tmp = data.copy()
    tmp['timestamp'] = pd.to_datetime(tmp['timestamp'], errors='coerce')
    tmp = tmp.dropna(subset=['timestamp']).sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    unique_ts = pd.DatetimeIndex(pd.unique(tmp['timestamp'])).sort_values()
    if len(unique_ts) < window_count:
        return [tmp]

    boundaries = np.linspace(0, len(unique_ts), window_count + 1, dtype=int)
    windows = []
    for i in range(window_count):
        left = boundaries[i]
        right = boundaries[i + 1]
        if right <= left:
            continue
        keep_ts = set(unique_ts[left:right])
        part = tmp[tmp['timestamp'].isin(keep_ts)].copy()
        if not part.empty:
            windows.append(part)
    return windows


def _evaluate_agent_on_validation(agent: DDQNAgent,
                                  eval_data: pd.DataFrame,
                                  config: dict,
                                  save_results: bool = False) -> dict:
    """在验证集上回测，返回用于选模的目标值。"""
    if eval_data is None or eval_data.empty:
        return {
            'score': -np.inf,
            'total_return': 0.0,
            'buy_hold_return': 0.0,
            'excess_return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'selection_passed': False,
            'oos_sharpes': []
        }

    data_config = config.get('data', {})
    training_config = config.get('training', {})
    backtest_config = config.get('backtest', {})
    initial_capital = backtest_config.get('initial_capital', 1000000)
    period_minutes = config.get('environment', {}).get('period_minutes', 20)
    bar_frequency = str(data_config.get('bar_frequency', 'minute')).lower()
    env_config = _build_env_config_from_config(config, initial_capital, period_minutes, bar_frequency)

    # <=0 表示使用完整验证集
    max_ts = int(training_config.get('validation_selection_max_timestamps', 0))
    eval_slice = _slice_recent_timestamps(eval_data, max_ts)
    if eval_slice is None or eval_slice.empty:
        return {
            'score': -np.inf,
            'total_return': 0.0,
            'buy_hold_return': 0.0,
            'excess_return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'selection_passed': False,
            'oos_sharpes': []
        }

    max_drawdown_gate = float(training_config.get('selection_max_drawdown', 1.0))
    oos_window_count = max(int(training_config.get('oos_window_count', 3)), 1)
    require_all_oos_positive_sharpe = bool(training_config.get('require_all_oos_positive_sharpe', True))
    oos_min_sharpe = float(training_config.get('oos_min_sharpe', 0.0))

    symbols = eval_slice['symbol'].astype(str).unique().tolist()
    backtest_engine = BacktestEngine(
        data_path=data_config.get('data_path', 'data/filtered'),
        symbols=symbols,
        initial_capital=initial_capital,
        period_minutes=period_minutes,
        bar_frequency=bar_frequency
    )
    result = backtest_engine.run_backtest(
        agent=agent,
        test_data=eval_slice,
        env_config=env_config,
        save_results=save_results,
        result_name=None
    )

    total_return = float(result.get('total_return', 0.0))
    metrics = result.get('metrics', {}) or {}
    sharpe = float(metrics.get('sharpe_ratio', 0.0))
    max_drawdown = float(metrics.get('max_drawdown', 0.0))
    buy_hold_return = float(result.get('benchmark_curves', {}).get('Buy & Hold', {}).get('final_return', 0.0))
    excess_return = total_return - buy_hold_return

    selection_passed = True
    gate_reasons = []

    # 门槛1：先筛最大回撤
    if max_drawdown_gate >= 0 and max_drawdown > max_drawdown_gate:
        selection_passed = False
        gate_reasons.append(f"max_drawdown>{max_drawdown_gate:.3f}")

    # 门槛2：至少3个OOS窗口都要正Sharpe
    oos_sharpes = []
    if require_all_oos_positive_sharpe:
        windows = _split_oos_windows(eval_slice, oos_window_count)
        if len(windows) < oos_window_count:
            selection_passed = False
            gate_reasons.append('insufficient_oos_windows')
        else:
            for idx, window_df in enumerate(windows, start=1):
                window_symbols = window_df['symbol'].astype(str).unique().tolist()
                window_engine = BacktestEngine(
                    data_path=data_config.get('data_path', 'data/filtered'),
                    symbols=window_symbols,
                    initial_capital=initial_capital,
                    period_minutes=period_minutes,
                    bar_frequency=bar_frequency
                )
                window_result = window_engine.run_backtest(
                    agent=agent,
                    test_data=window_df,
                    env_config=env_config,
                    save_results=False,
                    result_name=None
                )
                window_sharpe = float((window_result.get('metrics', {}) or {}).get('sharpe_ratio', 0.0))
                oos_sharpes.append(window_sharpe)
                if not np.isfinite(window_sharpe) or window_sharpe <= oos_min_sharpe:
                    selection_passed = False
                    gate_reasons.append(f"oos{idx}_sharpe<=min")

    selection_weight = float(training_config.get('selection_sharpe_weight', 0.2))
    raw_score = excess_return + selection_weight * sharpe
    score = raw_score if selection_passed else -np.inf

    if not selection_passed:
        print(f"  验证门槛未通过: {';'.join(gate_reasons)}")

    return {
        'score': float(score),
        'raw_score': float(raw_score),
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'excess_return': float(excess_return),
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'selection_passed': bool(selection_passed),
        'oos_sharpes': [float(x) for x in oos_sharpes]
    }


def _select_imitation_fixed_action(eval_data: pd.DataFrame, config: dict) -> int:
    """用验证集选择收益最好的固定动作，作为预热阶段教师策略。"""
    if eval_data is None or eval_data.empty:
        return 0

    data_config = config.get('data', {})
    backtest_config = config.get('backtest', {})
    initial_capital = backtest_config.get('initial_capital', 1000000)
    period_minutes = config.get('environment', {}).get('period_minutes', 20)
    bar_frequency = str(data_config.get('bar_frequency', 'minute')).lower()
    env_config = _build_env_config_from_config(config, initial_capital, period_minutes, bar_frequency)

    max_ts = int(config.get('training', {}).get('imitation_selection_max_timestamps', 0))
    teacher_data = _slice_recent_timestamps(eval_data, max_ts)

    symbols = teacher_data['symbol'].astype(str).unique().tolist()
    backtest_engine = BacktestEngine(
        data_path=data_config.get('data_path', 'data/filtered'),
        symbols=symbols,
        initial_capital=initial_capital,
        period_minutes=period_minutes,
        bar_frequency=bar_frequency
    )

    best_action = 0
    best_return = -np.inf
    for action_id in range(6):
        curve = backtest_engine._simulate_policy_curve(teacher_data, env_config, f'fixed_{action_id}')
        final_return = float(curve.get('final_return', -np.inf))
        if final_return > best_return:
            best_return = final_return
            best_action = action_id

    print(f"预热策略选择完成: fixed_{best_action} | 验证集收益率 {best_return*100:+.2f}%")
    return int(best_action)


def train_model(env: TradingEnvironment, agent: DDQNAgent, config: dict, val_data: pd.DataFrame = None) -> dict:
    """训练模型"""
    print("\n" + "="*80)
    print("开始训练")
    print("="*80)
    
    training_config = config.get('training', {})
    num_episodes = training_config.get('num_episodes', 1000)
    output_dir = training_config.get('output_dir', 'outputs/training')
    incremental_save_freq = training_config.get('incremental_save_freq', 10000)  # 每N步保存一次
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 用于存储每个episode的参数
    episode_data = []
    
    # 检查是否有增量checkpoint可以恢复
    resume_state = None
    latest_checkpoint = find_latest_incremental_checkpoint(output_dir)
    if latest_checkpoint:
        try:
            resume_state = load_incremental_checkpoint(agent, latest_checkpoint)
            print(f"\n将从Episode {resume_state['episode']}, Step {resume_state['step']}继续训练")
        except Exception as e:
            print(f"\n加载增量checkpoint失败: {e}")
            print("将从头开始训练")
            resume_state = None
    
    # 创建训练器
    trainer = Trainer(
        env=env,
        agent=agent,
        config=training_config,
        output_dir=output_dir
    )
    
    # 修改训练器以收集episode数据
    original_train = trainer.train
    
    def custom_train(num_episodes):
        """自定义训练函数，收集episode数据"""
        max_steps_per_episode = training_config.get('max_steps_per_episode', None)
        random_start = bool(training_config.get('random_start', False))
        use_best_checkpoint_for_eval = bool(training_config.get('use_best_checkpoint_for_eval', True))
        eval_freq = max(int(training_config.get('eval_freq', 10)), 1)
        selection_metric = str(training_config.get('model_selection_metric', 'train_return')).lower()
        imitation_episodes = max(int(training_config.get('imitation_episodes', 0)), 0)
        imitation_fixed_action = training_config.get('imitation_fixed_action', None)
        if imitation_episodes > 0 and imitation_fixed_action is None:
            imitation_fixed_action = _select_imitation_fixed_action(val_data, config)

        if imitation_episodes > 0:
            print(f"两阶段训练开启: Stage1固定动作预热 {imitation_episodes} 集, 动作={imitation_fixed_action}")
            print("Stage2进入DDQN微调")

        best_episode_return = -np.inf
        best_selection_score = -np.inf
        best_episode = 0
        best_selection_details = {}
        best_q_state = None
        best_target_state = None

        # 确定起始episode和步数
        start_episode = 1
        resume_step = 0
        resume_reward = 0
        resume_loss = 0
        resume_step_count = 0
        
        if resume_state:
            start_episode = resume_state['episode']
            resume_step = resume_state['step']
            resume_reward = resume_state['episode_reward']
            resume_loss = resume_state['episode_loss']
            resume_step_count = resume_state['step_count']
            print(f"\n从Episode {start_episode}, Step {resume_step}恢复训练")
        
        # 创建episode进度条
        episode_pbar = tqdm(range(start_episode, num_episodes + 1), 
                           desc="Training Episodes", 
                           unit="ep",
                           initial=start_episode-1,
                           total=num_episodes)
        
        for episode in episode_pbar:
            # 从checkpoint恢复时，直接开始新的episode（不需要快进）
            # checkpoint已经保存了网络权重、optimizer状态、epsilon等所有必要信息
            if episode == start_episode and resume_step > 0:
                print(f"\nEpisode {episode}/{num_episodes} 从checkpoint恢复...")
                print(f"  已恢复网络权重和训练状态")
                print(f"  之前在step {resume_step}保存，现在开始新的episode")
            
            # 重置环境，开始新的episode
            state = env.reset()
            if random_start and hasattr(env, 'max_steps') and env.max_steps > 200:
                # 训练时随机起点，避免每个episode重复同一段市场路径
                random_start_step = np.random.randint(20, max(21, env.max_steps - 100))
                env.current_step = random_start_step
                if hasattr(env, 'state_space'):
                    env.state_space.reset_history()
                state = env._get_observation()
            episode_reward = 0
            episode_loss = 0
            step_count = 0
            total_steps = 0
            
            if episode == start_episode and resume_step > 0:
                episode_pbar.set_postfix({"status": f"resumed ep{episode}"})
            else:
                episode_pbar.set_postfix({"status": f"running ep{episode}"})
            
            done = False
            while not done:
                # 选择动作
                if imitation_episodes > 0 and episode <= imitation_episodes and imitation_fixed_action is not None:
                    action = int(imitation_fixed_action)
                else:
                    action = agent.select_action(state)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 获取实际前进的步数（关键修复）
                actual_steps_advanced = info.get('steps_advanced', 1)
                
                # 检查state和reward是否有nan/inf（静默处理）
                if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                
                if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
                    next_state = np.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)
                
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0.0
                
                # 存储经验
                effective_action = int(info.get('effective_action', action))
                agent.store_transition(state, effective_action, reward, next_state, done)
                
                # 训练
                loss = agent.train()
                # 记录所有训练步骤，但只累加实际训练的loss
                if loss > 0:  # 实际进行了训练
                    episode_loss += loss
                    step_count += 1
                
                # 用于显示的当前loss（如果没有训练则显示最近的loss）
                if not hasattr(agent, 'last_valid_loss'):
                    agent.last_valid_loss = 0.0
                if loss > 0:
                    agent.last_valid_loss = loss
                
                episode_reward += reward
                state = next_state
                total_steps += 1  # 简化：每个环境step计数1步

                if max_steps_per_episode is not None and total_steps >= int(max_steps_per_episode):
                    done = True
                
                # 增量checkpoint保存
                if total_steps % incremental_save_freq == 0:
                    print(f"\n  保存增量checkpoint (步数: {total_steps})...")
                    save_incremental_checkpoint(
                        agent, episode, total_steps, 
                        episode_reward, episode_loss, step_count, output_dir
                    )
                
                # 每100步打印一次进度（显示单步奖励）
                if total_steps % 100 == 0:
                    # 显示当前有效的loss（如果当前步没有训练，显示最近的训练loss）
                    display_loss = loss if loss > 0 else getattr(agent, 'last_valid_loss', 0.0)
                    
                    # 计算不同类型的收益率
                    total_return = (env.portfolio_value - env.initial_capital) / env.initial_capital
                    
                    # 计算年化收益率（假设3年数据）
                    years = 3.0  # 2022-2024，3年数据
                    annualized_return = (1 + total_return) ** (1/years) - 1 if total_return > -1 else -1
                    
                    # 计算投资收益率（修复版：正确处理多头和空头）
                    current_position_value = 0
                    total_invested_cost = 0
                    
                    # 使用当前价格计算持仓价值
                    current_prices = env.price_cache.get(env.current_step, {})
                    for symbol, pos in env.positions.items():
                        if pos['quantity'] != 0 and symbol in current_prices:
                            current_price = current_prices[symbol].get('close', pos['avg_cost'])
                            # 持仓价值（可以为负，表示空头）
                            position_value = pos['quantity'] * current_price
                            current_position_value += position_value
                            
                            # 投资成本（多头为正，空头为负）
                            invested_cost = pos['quantity'] * pos['avg_cost']
                            total_invested_cost += invested_cost
                    
                    # 计算投资收益率
                    if abs(total_invested_cost) > 1:  # 避免除零和极小值
                        investment_return = (current_position_value - total_invested_cost) / abs(total_invested_cost)
                    else:
                        investment_return = 0.0
                    
                    # 获取策略名称
                    strategy_names = ['MarketNeutral', 'DualMA_MACD', 'BollingerBands', 'CTA', 'StatArb', 'LongShortEquity']
                    raw_action = int(info.get('raw_action', action))
                    effective_action = int(info.get('effective_action', raw_action))
                    raw_strategy_name = strategy_names[raw_action] if 0 <= raw_action < len(strategy_names) else f'Strategy_{raw_action}'
                    effective_strategy_name = strategy_names[effective_action] if 0 <= effective_action < len(strategy_names) else f'Strategy_{effective_action}'
                    lock_flag = 'ON' if info.get('profit_lock_active', False) else 'OFF'
                    
                    # 显示更详细的收益率信息
                    print(f"  Step {total_steps:5d} | 单步奖励: {reward:+7.4f} | 累计收益率: {total_return*100:+6.2f}% | 年化收益率: {annualized_return*100:+5.2f}% | 组合价值: {env.portfolio_value:9,.0f} | 现金: {env.cash:8,.0f} | 动作(raw->eff): {raw_strategy_name:12s}->{effective_strategy_name:12s} | 锁盈: {lock_flag} | Loss: {display_loss:.4f}")
            
            # 计算平均loss
            avg_loss = episode_loss / step_count if step_count > 0 else 0
            
            # 计算累计收益率（从第1个episode到当前episode）
            total_return = (env.portfolio_value - env.initial_capital) / env.initial_capital
            
            # 计算投资收益率（修复版：正确处理多头和空头）
            current_position_value = 0
            total_invested_cost = 0
            
            # 使用当前价格计算持仓价值
            current_prices = env.price_cache.get(env.current_step, {})
            for symbol, pos in env.positions.items():
                if pos['quantity'] != 0 and symbol in current_prices:
                    current_price = current_prices[symbol].get('close', pos['avg_cost'])
                    # 持仓价值（可以为负，表示空头）
                    position_value = pos['quantity'] * current_price
                    current_position_value += position_value
                    
                    # 投资成本（多头为正，空头为负）
                    invested_cost = pos['quantity'] * pos['avg_cost']
                    total_invested_cost += invested_cost
            
            # 计算投资收益率
            if abs(total_invested_cost) > 1:  # 避免除零和极小值
                investment_return = (current_position_value - total_invested_cost) / abs(total_invested_cost)
            else:
                investment_return = 0.0
            
            invested_capital = env.initial_capital - env.cash
            
            # 收集episode参数
            episode_params = {
                'episode': episode,
                'reward': float(episode_reward),
                'loss': float(avg_loss),
                'total_return': float(total_return),
                'investment_return': float(investment_return),
                'epsilon': float(agent.epsilon),
                'portfolio_value': float(env.portfolio_value),
                'cash': float(env.cash),
                'invested_capital': float(invested_capital),
                'position_value': float(current_position_value),
                'invested_cost': float(total_invested_cost),
                'steps': step_count
            }
            episode_data.append(episode_params)

            # 记录训练过程中的最高训练收益（用于诊断）
            if total_return > best_episode_return:
                best_episode_return = float(total_return)

            # 选模分数：默认训练收益；可切换为验证集“超额收益+夏普”
            selection_score = float(total_return)
            selection_details = {
                'train_return': float(total_return),
                'metric': 'train_return'
            }
            if selection_metric == 'val_excess_sharpe' and val_data is not None and not val_data.empty:
                if (episode % eval_freq == 0) or (episode == num_episodes):
                    val_eval = _evaluate_agent_on_validation(agent, val_data, config, save_results=False)
                    selection_score = float(val_eval.get('score', -np.inf))
                    selection_details = {
                        'metric': 'val_excess_sharpe',
                        'val_total_return': float(val_eval.get('total_return', 0.0)),
                        'val_buy_hold_return': float(val_eval.get('buy_hold_return', 0.0)),
                        'val_excess_return': float(val_eval.get('excess_return', 0.0)),
                        'val_sharpe': float(val_eval.get('sharpe', 0.0)),
                        'val_max_drawdown': float(val_eval.get('max_drawdown', 0.0)),
                        'selection_passed': bool(val_eval.get('selection_passed', True)),
                        'oos_sharpes': list(val_eval.get('oos_sharpes', [])),
                        'score': selection_score
                    }
                    oos_str = ','.join([f"{float(x):+.2f}" for x in selection_details['oos_sharpes']])
                    print(
                        "  验证集选模: "
                        f"超额收益 {selection_details['val_excess_return']*100:+.2f}% | "
                        f"Sharpe {selection_details['val_sharpe']:+.3f} | "
                        f"MDD {selection_details['val_max_drawdown']*100:.2f}% | "
                        f"Pass {selection_details['selection_passed']} | "
                        f"OOS[{oos_str}] | "
                        f"Score {selection_score:+.4f}"
                    )
                else:
                    selection_score = -np.inf

            if selection_score > best_selection_score:
                best_selection_score = float(selection_score)
                best_episode = episode
                best_selection_details = dict(selection_details)
                best_q_state = {
                    k: v.detach().cpu().clone()
                    for k, v in agent.q_network.state_dict().items()
                }
                best_target_state = {
                    k: v.detach().cpu().clone()
                    for k, v in agent.target_network.state_dict().items()
                }
                print(
                    f"  更新最佳权重: Episode {best_episode} | "
                    f"选模指标 {best_selection_details.get('metric', 'unknown')} | "
                    f"Score {best_selection_score:+.4f}"
                )
            
            # 衰减epsilon（预热期不衰减，切入RL微调时重置）
            if imitation_episodes > 0 and episode == imitation_episodes:
                agent.epsilon = float(config.get('agent', {}).get('epsilon_start', agent.epsilon))
                print(f"  预热结束，epsilon重置为: {agent.epsilon:.3f}")
            elif episode > imitation_episodes:
                agent.decay_epsilon()
            
            # 更新目标网络（每N个episodes）
            if episode % agent.target_update_freq == 0:
                agent.update_target_network()
                print(f"  目标网络已更新（Episode {episode}）")
            
            # 更新episode计数
            agent.episode_count = episode
            
            # 更新进度条信息
            episode_pbar.set_postfix({
                "return": f"{total_return*100:+.2f}%",
                "value": f"{env.portfolio_value:,.0f}",
                "epsilon": f"{agent.epsilon:.3f}"
            })
            
            # 详细信息（每5个episode显示一次）
            if episode % 5 == 0:
                # 计算策略使用统计
                episode_history = env.get_episode_history()
                strategy_counts = {}
                for step_info in episode_history:
                    strategy = step_info.get('strategy', 'Unknown')
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                
                # 找出使用最多的策略
                if strategy_counts:
                    most_used_strategy = max(strategy_counts, key=strategy_counts.get)
                    strategy_diversity = len(strategy_counts)
                    print(f"  主要策略: {most_used_strategy} | 策略多样性: {strategy_diversity}/6 | 平均Loss: {avg_loss:.4f}")
                else:
                    print(f"  平均Loss: {avg_loss:.4f}")
            
            # 保存模型
            if episode % training_config.get('save_freq', 50) == 0:
                model_file = os.path.join(output_dir, f'model_episode_{episode}.pth')
                agent.save(model_file)
            
            # Episode结束后清理该episode的增量checkpoints
            checkpoint_dir = os.path.join(output_dir, 'incremental_checkpoints')
            if os.path.exists(checkpoint_dir):
                for f in os.listdir(checkpoint_dir):
                    if f.startswith(f'checkpoint_ep{episode}_'):
                        try:
                            os.remove(os.path.join(checkpoint_dir, f))
                        except:
                            pass

        if use_best_checkpoint_for_eval and best_q_state is not None and best_target_state is not None:
            agent.q_network.load_state_dict(best_q_state)
            agent.target_network.load_state_dict(best_target_state)
            print(
                f"\n已回滚到最佳权重: Episode {best_episode} | "
                f"选模分数 {best_selection_score:+.4f} | "
                f"选模方式 {best_selection_details.get('metric', 'unknown')}"
            )

        return {
            'total_episodes': num_episodes,
            'avg_reward': np.mean([d['reward'] for d in episode_data]),
            'final_portfolio_value': episode_data[-1]['portfolio_value'],
            'training_time': 0,
            'best_episode': int(best_episode),
            'best_episode_return': float(best_episode_return if np.isfinite(best_episode_return) else 0.0),
            'best_selection_score': float(best_selection_score if np.isfinite(best_selection_score) else 0.0),
            'best_selection_details': best_selection_details
        }
    
    # 替换训练函数
    trainer.train = custom_train
    
    # 开始训练
    training_stats = trainer.train(num_episodes=num_episodes)
    
    # 保存episode参数到JSON
    save_episode_parameters(episode_data, output_dir)
    
    # 生成可视化图表
    plot_training_metrics(episode_data, output_dir)
    
    print("\n" + "="*80)
    print("训练完成")
    print("="*80)
    print(f"总episode数: {training_stats.get('total_episodes', 0)}")
    print(f"平均奖励: {training_stats.get('avg_reward', 0):.2f}")
    print(f"最终组合价值: {training_stats.get('final_portfolio_value', 0):,.2f}")
    if training_stats.get('best_episode', 0):
        print(
            f"最佳episode: {training_stats.get('best_episode')} | "
            f"最佳训练收益率: {training_stats.get('best_episode_return', 0)*100:+.2f}% | "
            f"最佳选模分数: {training_stats.get('best_selection_score', 0):+.4f}"
        )
    print(f"训练时间: {training_stats.get('training_time', 0):.2f}秒")
    
    return training_stats


def evaluate_model(env: TradingEnvironment, agent: DDQNAgent, config: dict):
    """评估模型"""
    print("\n" + "="*80)
    print("模型评估")
    print("="*80)
    
    eval_config = config.get('evaluation', {})
    num_episodes = eval_config.get('num_episodes', 10)
    output_dir = eval_config.get('output_dir', 'outputs/evaluation')
    
    # 创建评估器
    evaluator = Evaluator(env, agent)
    
    # 运行评估
    evaluation_results = evaluator.evaluate(
        num_episodes=num_episodes,
        render=False,
        save_results=True,
        output_dir=output_dir
    )
    
    return evaluation_results


def run_backtest(agent: DDQNAgent, test_data: pd.DataFrame, config: dict):
    """运行回测"""
    print("\n" + "="*80)
    print("运行回测")
    print("="*80)
    
    # 检查测试数据
    if test_data.empty:
        print("❌ 测试数据为空，跳过回测")
        return {
            'initial_capital': 1000000,
            'final_portfolio_value': 1000000,
            'total_return': 0.0,
            'metrics': {'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}
        }
    
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试数据列: {test_data.columns.tolist()}")
    
    backtest_config = config.get('backtest', {})
    data_config = config.get('data', {})
    env_config = config.get('environment', {})
    
    # 检查测试数据中的股票数量
    if 'symbol' not in test_data.columns:
        print("❌ 测试数据中没有'symbol'列，跳过回测")
        return {
            'initial_capital': 1000000,
            'final_portfolio_value': 1000000,
            'total_return': 0.0,
            'metrics': {'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}
        }
    
    test_symbols = test_data['symbol'].unique()
    config_symbols = resolve_symbols(data_config)
    
    print(f"配置中的股票: {len(config_symbols)} 只")
    print(f"测试数据中的股票: {len(test_symbols)} 只")
    
    if len(test_symbols) == 0:
        print("❌ 测试数据中没有股票，跳过回测")
        return {
            'initial_capital': 1000000,
            'final_portfolio_value': 1000000,
            'total_return': 0.0,
            'metrics': {'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}
        }
    
    # 如果股票数量不匹配，过滤测试数据
    if len(test_symbols) != len(config_symbols):
        print(f"警告: 股票数量不匹配，过滤测试数据...")
        # 只保留配置中指定的股票
        available_symbols = [s for s in config_symbols if s in test_symbols]
        if len(available_symbols) < len(config_symbols):
            print(f"警告: 测试数据中只有 {len(available_symbols)} 只股票可用")
            print(f"可用股票: {available_symbols}")
        
        if len(available_symbols) == 0:
            print("❌ 没有可用的股票进行回测，跳过回测")
            return {
                'initial_capital': 1000000,
                'final_portfolio_value': 1000000,
                'total_return': 0.0,
                'metrics': {'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}
            }
        
        # 过滤测试数据
        test_data = test_data[test_data['symbol'].isin(available_symbols)].copy()
        print(f"过滤后测试数据: {len(test_data)} 行")
    
    data_path = data_config.get('data_path', 'data/filtered')
    # 使用实际存在于测试数据中的股票代码
    symbols = test_data['symbol'].unique().tolist()
    initial_capital = backtest_config.get('initial_capital', 1000000)
    period_minutes = env_config.get('period_minutes', 20)
    bar_frequency = str(data_config.get('bar_frequency', 'minute')).lower()
    
    print(f"回测将使用 {len(symbols)} 只股票: {symbols}")
    
    try:
        # 创建回测引擎
        backtest_engine = BacktestEngine(
            data_path=data_path,
            symbols=symbols,
            initial_capital=initial_capital,
            period_minutes=period_minutes,
            bar_frequency=bar_frequency
        )
        
        # 运行回测
        backtest_env_config = _build_env_config_from_config(config, initial_capital, period_minutes, bar_frequency)
        backtest_result = backtest_engine.run_backtest(
            agent=agent,
            test_data=test_data,
            env_config=backtest_env_config,
            save_results=True,
            result_name=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # 打印回测摘要
        print("\n回测结果摘要:")
        print(f"  初始资金: {backtest_result['initial_capital']:,.2f}")
        print(f"  最终资金: {backtest_result['final_portfolio_value']:,.2f}")
        print(f"  总收益率: {backtest_result['total_return']*100:.2f}%")
        print(f"  夏普比率: {backtest_result['metrics'].get('sharpe_ratio', 0):.3f}")
        print(f"  最大回撤: {backtest_result['metrics'].get('max_drawdown', 0)*100:.2f}%")
        print(f"  胜率: {backtest_result['metrics'].get('win_rate', 0)*100:.2f}%")
        
        return backtest_result
        
    except Exception as e:
        print(f"❌ 回测执行失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回默认结果
        return {
            'initial_capital': 1000000,
            'final_portfolio_value': 1000000,
            'total_return': 0.0,
            'metrics': {'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}
        }


def save_model(agent: DDQNAgent, config: dict):
    """保存模型"""
    print("\n" + "="*80)
    print("保存模型")
    print("="*80)
    
    model_config = config.get('model', {})
    model_path = model_config.get('model_path', 'models')
    
    # 创建模型目录
    os.makedirs(model_path, exist_ok=True)
    
    # 生成模型文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = os.path.join(model_path, f"ddqn_model_{timestamp}.pth")
    
    # 保存模型
    agent.save(model_file)
    
    print(f"模型已保存到: {model_file}")
    
    return model_file


def main():
    """主函数"""
    print("="*80)
    print("多策略强化学习交易系统 - 训练脚本")
    print("="*80)
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='训练DDQN交易智能体')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='配置文件路径（默认: configs/training_config.yaml）')
    parser.add_argument('--resume', type=str, default=None, 
                       help='从指定的模型文件继续训练（例如：models/ddqn_model_20260304_123456.pth）')
    parser.add_argument('--resume-latest', action='store_true',
                       help='从最新的模型文件继续训练')
    args = parser.parse_args()
    
    try:
        # 1. 加载配置
        print("\n[1/7] 加载配置...")
        config = load_config(args.config)
        print("配置加载成功")
        
        # 确定是否需要恢复训练
        resume_from = None
        if args.resume:
            resume_from = args.resume
            print(f"\n将从指定模型恢复训练: {resume_from}")
        elif args.resume_latest:
            # 查找最新的模型文件
            model_path = config.get('model', {}).get('model_path', 'models')
            model_files = []
            if os.path.exists(model_path):
                for f in os.listdir(model_path):
                    if f.startswith('ddqn_model_') and f.endswith('.pth'):
                        full_path = os.path.join(model_path, f)
                        model_files.append((full_path, os.path.getmtime(full_path)))
            
            if model_files:
                # 按修改时间排序，取最新的
                model_files.sort(key=lambda x: x[1], reverse=True)
                resume_from = model_files[0][0]
                print(f"\n找到最新模型: {resume_from}")
            else:
                print("\n未找到可恢复的模型，将从头开始训练")
        
        # 2. 准备数据
        print("\n[2/7] 准备数据...")
        data_dict = prepare_data(config)
        train_data = data_dict['train']
        val_data = data_dict['val']
        test_data = data_dict['test']
        
        # 3. 创建环境
        print("\n[3/7] 创建环境...")
        env = create_environment(train_data, config)
        
        # 4. 创建智能体（可能从检查点恢复）
        print("\n[4/7] 创建智能体...")
        agent = create_agent(env, config, resume_from=resume_from)
        
        # 5. 训练模型
        print("\n[5/7] 训练模型...")
        training_stats = train_model(env, agent, config, val_data=val_data)
        
        # 6. 评估模型
        print("\n[6/7] 评估模型...")
        evaluation_results = evaluate_model(env, agent, config)
        
        # 7. 运行回测
        print("\n[7/7] 运行回测...")
        backtest_result = run_backtest(agent, test_data, config)
        
        # 8. 保存模型
        model_file = save_model(agent, config)
        
        print("\n" + "="*80)
        print("训练流程完成!")
        print("="*80)
        print(f"训练统计: {training_stats.get('total_episodes', 0)} episodes")
        print(f"模型文件: {model_file}")
        print(f"回测收益率: {backtest_result['total_return']*100:.2f}%")
        print("="*80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
