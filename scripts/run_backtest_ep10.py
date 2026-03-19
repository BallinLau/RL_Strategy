#!/usr/bin/env python3
"""
回测脚本 - Episode 10 模型回测
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import load_config, prepare_data, create_environment, create_agent, run_backtest
import json


def main():
    config_path = "configs/training_config_crsp_daily_v3_fullval_oos3_short_gpu.yaml"
    pth_path = "outputs/training_crsp_daily_v3_fullval_oos3_short_gpu/model_episode_10.pth"
    
    print("="*80)
    print("Episode 10 模型回测")
    print("="*80)
    print(f"配置: {config_path}")
    print(f"模型: {pth_path}")
    print("="*80)
    
    # 加载配置
    print("\n[1/5] 加载配置...")
    cfg = load_config(config_path)
    
    # 准备数据
    print("\n[2/5] 准备数据...")
    data_dict = prepare_data(cfg)
    train_data = data_dict["train"]
    test_data = data_dict["test"]
    
    # 创建环境
    print("\n[3/5] 创建交易环境...")
    env = create_environment(train_data, cfg)
    
    # 创建 agent 并加载模型
    print("\n[4/5] 加载模型...")
    agent = create_agent(env, cfg, resume_from=pth_path)
    
    # 运行回测
    print("\n[5/5] 运行回测...")
    print("="*80)
    res = run_backtest(agent, test_data, cfg)
    
    # 打印结果
    print("\n" + "="*80)
    print("=== Backtest Summary ===")
    print("="*80)
    print(f"Total Return: {res.get('total_return', 0):.4f} ({res.get('total_return', 0)*100:.2f}%)")
    print(f"Sharpe Ratio: {res.get('metrics', {}).get('sharpe_ratio', 0):.4f}")
    print(f"Max Drawdown: {res.get('metrics', {}).get('max_drawdown', 0):.4f} ({res.get('metrics', {}).get('max_drawdown', 0)*100:.2f}%)")
    print(f"Win Rate: {res.get('metrics', {}).get('win_rate', 0):.4f} ({res.get('metrics', {}).get('win_rate', 0)*100:.2f}%)")
    
    # 保存结果到 backtest/ep10
    print("\n保存结果...")
    os.makedirs("backtest/ep10", exist_ok=True)
    
    # 保存 JSON 结果
    with open("backtest/ep10/backtest_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    
    # 保存文本报告
    with open("backtest/ep10/backtest_report.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("Episode 10 回测报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"模型路径: {pth_path}\n")
        f.write(f"配置路径: {config_path}\n\n")
        f.write("="*80 + "\n")
        f.write("性能指标\n")
        f.write("="*80 + "\n")
        f.write(f"总收益率: {res.get('total_return', 0)*100:.2f}%\n")
        f.write(f"夏普比率: {res.get('metrics', {}).get('sharpe_ratio', 0):.4f}\n")
        f.write(f"最大回撤: {res.get('metrics', {}).get('max_drawdown', 0)*100:.2f}%\n")
        f.write(f"胜率: {res.get('metrics', {}).get('win_rate', 0)*100:.2f}%\n")
        f.write(f"初始资金: {res.get('initial_capital', 0):,.2f}\n")
        f.write(f"最终资金: {res.get('final_capital', 0):,.2f}\n")
    
    print("结果已保存到:")
    print("  - backtest/ep10/backtest_results.json")
    print("  - backtest/ep10/backtest_report.txt")
    print("="*80)


if __name__ == '__main__':
    main()
