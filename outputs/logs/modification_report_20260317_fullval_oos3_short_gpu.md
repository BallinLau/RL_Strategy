# 修改报告（2026-03-17）

## 目标与范围
- 目标：按你的要求把训练改为“完整验证集选模 + 回撤门槛 + 3个OOS窗口正Sharpe门槛”，并给出允许做空、奖励瘦身、降交易频率、GPU训练配置。
- 代码变更文件：
  - `scripts/train.py`
  - `configs/training_config_crsp_daily_v3_fullval_oos3_short_gpu.yaml`（新增）

## 逐文件逐行变更

### 1) `scripts/train.py`

1. `create_agent` 增加 GPU 安全回退  
   - 位置：`scripts/train.py:401-404, 417`
   - 变更：
     - 读取 `agent.device` 到 `requested_device`
     - 若配置是 `cuda*` 但当前 `torch.cuda.is_available()` 为 `False`，自动回退 `cpu`
     - `DDQNAgent(..., device=requested_device)` 使用回退后的设备
   - 原因：保证命令直接可跑，不因无CUDA直接报错退出。

2. 新增 OOS 切窗函数  
   - 位置：`scripts/train.py:643-669`
   - 变更：
     - 新增 `_split_oos_windows(data, window_count)`，按时间顺序把验证集切成多个窗口
   - 原因：用于“至少3个OOS窗口都为正Sharpe”的协议门槛。

3. 验证集选模逻辑重构（完整验证集 + 门槛筛选）  
   - 位置：`scripts/train.py:672-795`
   - 变更：
     - `validation_selection_max_timestamps` 默认值改为 `0`（`<=0` 即完整验证集）
     - 新增门槛参数读取：
       - `selection_max_drawdown`（回撤门槛）
       - `oos_window_count`
       - `require_all_oos_positive_sharpe`
       - `oos_min_sharpe`
     - 门槛1：先筛最大回撤（`max_drawdown > selection_max_drawdown` 则不通过）
     - 门槛2：按OOS窗口逐窗回测，若任一窗 `Sharpe <= oos_min_sharpe` 则不通过
     - 评分保持：`raw_score = excess_return + selection_sharpe_weight * sharpe`
     - 若门槛不通过，`score = -inf`
     - 返回字段新增：
       - `raw_score`
       - `max_drawdown`
       - `selection_passed`
       - `oos_sharpes`
   - 原因：实现你指定的“先风险门槛，再收益/Sharpe比较”的选模规则。

4. 预热教师策略改为可用完整验证集  
   - 位置：`scripts/train.py:810-811`
   - 变更：
     - `imitation_selection_max_timestamps` 默认值改为 `0`（完整验证集）
   - 原因：避免教师动作只在局部样本上选优，减少偏置。

5. 训练日志输出补充门槛信息  
   - 位置：`scripts/train.py:1105-1124`
   - 变更：
     - 验证打印加入 `MDD`、`Pass`、`OOS[sharpe列表]`
   - 原因：快速判断“分数好但未过门槛”的情况，便于诊断。

### 2) `configs/training_config_crsp_daily_v3_fullval_oos3_short_gpu.yaml`（新增）

1. 数据与切分  
   - 位置：`...yaml:1-12`
   - 设置：
     - `bar_frequency: daily`
     - `data_path: .../data/crsp_ohlc_all`
     - `split_mode: calendar_year`
     - `test_year: 2025`
   - 作用：2025年作测试，其他年份用于训练/验证。

2. 完整验证集选模 + OOS门槛  
   - 位置：`...yaml:23-31`
   - 设置：
     - `model_selection_metric: val_excess_sharpe`
     - `validation_selection_max_timestamps: 0`
     - `imitation_selection_max_timestamps: 0`
     - `selection_max_drawdown: 0.18`
     - `oos_window_count: 3`
     - `require_all_oos_positive_sharpe: true`
     - `oos_min_sharpe: 0.0`

3. 允许做空与降交易频率  
   - 位置：`...yaml:40-57`
   - 设置：
     - `allow_short: true`
     - `rebalance_tolerance: 0.02`
     - `min_signal_threshold: 0.05`
     - `min_action_hold_steps: 8`
     - `max_trade_fraction: 0.03`
     - `profit_lock_cooldown_steps: 10`

4. 奖励瘦身（excess主导，惩罚弱正则）  
   - 位置：`...yaml:100-106`
   - 设置：
     - `reward_mode: excess_dominant`
     - `excess_reward_weight: 1.0`
     - `terminal_return_bonus: 1.5`
     - `drawdown_penalty_weight: 0.15`
     - `turnover_penalty_weight: 0.01`
     - `switch_penalty_weight: 0.1`
     - `directional_hit_weight: 0.03`

5. GPU配置  
   - 位置：`...yaml:67, 118-120`
   - 设置：
     - `agent.device: cuda`
     - `device.use_cuda: true`
     - `device.cuda_device: 0`

## 验证
- 已做语法检查：
  - `python3 -m py_compile scripts/train.py` 通过。

## 运行命令（GPU）
```bash
cd "/home/fit/zhuyingz/WORK/LiuHao/RL_Strategy" && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
/Users/ballinliu/anaconda3/bin/python scripts/train.py \
--config configs/training_config_crsp_daily_v3_fullval_oos3_short_gpu.yaml
```
