#!/usr/bin/env python3
"""
回测脚本 V2 - 使用更新后的配置文件
基于 training_config_crsp_daily_v3_fullval_oos3_short_gpu.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from scripts.train import load_config, prepare_data, create_environment, create_agent, run_backtest


def parse_args():
    parser = argparse.ArgumentParser(description='运行模型回测 V2')
    parser.add_argument('--episode', type=int, default=None, 
                        help='Episode 编号 (例如: 10, 20, 30)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型文件路径（覆盖 episode 参数）')
    parser.add_argument('--config-path', type=str, 
                        default='configs/training_config_crsp_daily_v3_fullval_oos3_short_gpu.yaml',
                        help='配置文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（覆盖默认目录）')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 确定模型路径和输出目录
    if args.model_path:
        pth_path = args.model_path
        if args.output_dir:
            output_dir = args.output_dir
        else:
            import re
            match = re.search(r'episode_(\d+)', pth_path)
            if match:
                output_dir = f"backtest_v2/ep{match.group(1)}"
            else:
                output_dir = "backtest_v2/custom"
    elif args.episode is not None:
        pth_path = f"outputs/training_crsp_daily_v3_fullval_oos3_short_gpu/model_episode_{args.episode}.pth"
        output_dir = args.output_dir or f"backtest_v2/ep{args.episode}"
    else:
        print("错误: 请指定 --episode 或 --model-path")
        sys.exit(1)
    
    config_path = args.config_path
    
    print("="*80)
    print(f"回测 V2 - Episode {args.episode if args.episode else 'Custom'}")
    print("="*80)
    print(f"配置: {config_path}")
    print(f"模型: {pth_path}")
    print(f"输出: {output_dir}")
    print("="*80)
    
    # 加载配置
    print("\n[1/5] 加载配置...")
    cfg = load_config(config_path)
    
    # 显示关键配置
    print(f"  训练截止: {cfg.get('data', {}).get('train_end_date', 'N/A')}")
    print(f"  测试年份: {cfg.get('data', {}).get('test_year', 'N/A')}")
    print(f"  交易成本: {cfg.get('environment', {}).get('transaction_cost', 'N/A')}")
    print(f"  滑点: {cfg.get('environment', {}).get('slippage_rate', 'N/A')}")
    
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
    
    # 保存结果
    print("\n保存结果...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 JSON 结果
    json_path = os.path.join(output_dir, 'backtest_results.json')
    with open(json_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    
    # 保存文本报告
    report_path = os.path.join(output_dir, 'backtest_report.txt')
    with open(report_path, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"回测 V2 - Episode {args.episode if args.episode else 'Custom'} 报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"模型路径: {pth_path}\n")
        f.write(f"配置路径: {config_path}\n")
        f.write(f"训练截止: {cfg.get('data', {}).get('train_end_date', 'N/A')}\n")
        f.write(f"测试年份: {cfg.get('data', {}).get('test_year', 'N/A')}\n")
        f.write(f"交易成本: {cfg.get('environment', {}).get('transaction_cost', 'N/A')}\n\n")
        f.write("="*80 + "\n")
        f.write("性能指标\n")
        f.write("="*80 + "\n")
        f.write(f"总收益率: {res.get('total_return', 0)*100:.2f}%\n")
        f.write(f"夏普比率: {res.get('metrics', {}).get('sharpe_ratio', 0):.4f}\n")
        f.write(f"最大回撤: {res.get('metrics', {}).get('max_drawdown', 0)*100:.2f}%\n")
        f.write(f"胜率: {res.get('metrics', {}).get('win_rate', 0)*100:.2f}%\n")
        f.write(f"初始资金: {res.get('initial_capital', 0):,.2f}\n")
        f.write(f"最终资金: {res.get('final_capital', 0):,.2f}\n")
    
    print(f"\n结果已保存到:")
    print(f"  - {json_path}")
    print(f"  - {report_path}")
    print("="*80)


if __name__ == '__main__':
    main()
