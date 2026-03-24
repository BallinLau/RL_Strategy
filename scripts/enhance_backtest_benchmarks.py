#!/usr/bin/env python3
"""
Enhance backtest JSON with action-strategy benchmarks and future-window oracle benchmark.
"""

import argparse
import copy
import json
import math
import os
import shutil
import sys
from typing import Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.train import load_config, prepare_data  # noqa: E402
from src.training.backtest import BacktestEngine  # noqa: E402
from src.strategies.strategy_manager import StrategyManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance backtest benchmarks and plots")
    parser.add_argument(
        "--episode-dir",
        type=str,
        required=True,
        help="Episode backtest directory, e.g. backtest_v2/ep100",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/training_config_crsp_daily_v3_fullval_oos3_short_gpu.yaml",
        help="Config path used to load the test dataset",
    )
    parser.add_argument(
        "--future-window-steps",
        type=int,
        default=1,
        help="Lookahead window in decision steps for oracle best strategy",
    )
    return parser.parse_args()


def load_backtest_json(episode_dir: str) -> Tuple[str, Dict]:
    json_path = os.path.join(episode_dir, "backtest_results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json_path, json.load(f)


def backup_json(json_path: str) -> None:
    backup_path = json_path.replace(".json", ".before_benchmark_enhancement.json")
    if not os.path.exists(backup_path):
        shutil.copy2(json_path, backup_path)


def to_datetime_index(market_timestamps: List[str]) -> pd.DatetimeIndex:
    if not market_timestamps:
        return pd.DatetimeIndex([])
    ts = pd.to_datetime(pd.Series(market_timestamps), errors="coerce")
    return pd.DatetimeIndex(ts.dropna())


def strategy_name_map() -> Dict[int, str]:
    manager = StrategyManager()
    return {i: manager.get_strategy_name(i) for i in range(len(manager))}


def infer_periods_per_year(timestamps: List[str], n_steps: int) -> float:
    dt_index = to_datetime_index(timestamps)
    if len(dt_index) >= 2:
        span_years = (dt_index[-1] - dt_index[0]).total_seconds() / (365.25 * 24 * 3600)
        if span_years > 0:
            return max(float(n_steps) / span_years, 1.0)
    return 252.0


def curve_metrics(
    cumulative_returns: List[float],
    market_timestamps: List[str],
    initial_capital: float,
    actions: List[int] = None,
) -> Dict[str, float]:
    if not cumulative_returns:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "final_portfolio_value": initial_capital,
            "switch_count": 0,
        }

    wealth = 1.0 + np.asarray(cumulative_returns, dtype=float)
    wealth = np.where(np.isfinite(wealth), wealth, np.nan)
    wealth = np.clip(wealth, 1e-12, None)
    wealth = pd.Series(wealth).ffill().bfill().to_numpy()

    wealth_with_base = np.concatenate([[1.0], wealth])
    step_returns = wealth_with_base[1:] / wealth_with_base[:-1] - 1.0
    total_return = float(wealth[-1] - 1.0)

    periods_per_year = infer_periods_per_year(market_timestamps, len(step_returns))
    dt_index = to_datetime_index(market_timestamps)
    if len(dt_index) >= 2:
        years = (dt_index[-1] - dt_index[0]).total_seconds() / (365.25 * 24 * 3600)
        annualized_return = float(wealth[-1] ** (1.0 / years) - 1.0) if years > 0 else total_return
    else:
        annualized_return = float(wealth[-1] ** (periods_per_year / max(len(step_returns), 1)) - 1.0)

    volatility = float(np.std(step_returns) * math.sqrt(periods_per_year)) if len(step_returns) > 1 else 0.0
    sharpe_ratio = float(annualized_return / volatility) if volatility > 0 else 0.0

    running_peak = np.maximum.accumulate(wealth)
    drawdown = (running_peak - wealth) / np.clip(running_peak, 1e-12, None)
    max_drawdown = float(np.max(drawdown)) if len(drawdown) else 0.0
    calmar_ratio = float(annualized_return / max_drawdown) if max_drawdown > 0 else 0.0
    win_rate = float(np.mean(step_returns > 0)) if len(step_returns) else 0.0

    switch_count = 0
    if actions:
        switch_count = int(sum(1 for i in range(1, len(actions)) if int(actions[i]) != int(actions[i - 1])))

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "final_portfolio_value": float(initial_capital * wealth[-1]),
        "switch_count": switch_count,
    }


def simulate_fixed_action_benchmarks(
    test_data: pd.DataFrame,
    env_config: Dict,
    data_path: str,
    bar_frequency: str,
) -> Dict[str, Dict]:
    symbols = test_data["symbol"].astype(str).unique().tolist()
    engine = BacktestEngine(
        data_path=data_path,
        symbols=symbols,
        initial_capital=env_config["initial_capital"],
        period_minutes=env_config["period_minutes"],
        bar_frequency=bar_frequency,
    )
    action_names = strategy_name_map()
    results = {}
    for action_id, strategy_name in action_names.items():
        curve = engine._simulate_policy_curve(test_data, env_config, f"fixed_{action_id}")
        results[strategy_name] = {
            "strategy_id": int(action_id),
            "policy_name": f"fixed_{action_id}",
            "actions_history": curve.get("actions_history", []),
            "market_timestamps": curve.get("market_timestamps", []),
            "portfolio_values": curve.get("portfolio_values", []),
            "cumulative_returns": curve.get("cumulative_returns", []),
            "final_return": float(curve.get("final_return", 0.0)),
        }
    return results


def choose_best_action_with_lookahead(env, action_ids: List[int], future_window_steps: int) -> int:
    scores = []
    for action_id in action_ids:
        sim_env = copy.deepcopy(env)
        start_value = float(sim_env.portfolio_value)
        last_value = start_value
        for _ in range(max(int(future_window_steps), 1)):
            _, _, sim_done, sim_info = sim_env.step(int(action_id))
            last_value = float(sim_info.get("portfolio_value", sim_env.portfolio_value))
            if sim_done:
                break
        score = (last_value / start_value - 1.0) if start_value > 0 else -np.inf
        scores.append((float(score), int(action_id)))
    scores.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return int(scores[0][1])


def simulate_future_best_strategy(
    test_data: pd.DataFrame,
    env_config: Dict,
    data_path: str,
    bar_frequency: str,
    future_window_steps: int,
) -> Dict:
    symbols = test_data["symbol"].astype(str).unique().tolist()
    engine = BacktestEngine(
        data_path=data_path,
        symbols=symbols,
        initial_capital=env_config["initial_capital"],
        period_minutes=env_config["period_minutes"],
        bar_frequency=bar_frequency,
    )
    env = engine._create_backtest_env(test_data, env_config)
    action_names = strategy_name_map()
    action_ids = sorted(action_names.keys())

    env.reset()
    done = False
    portfolio_values = []
    actions_history = []
    strategy_names_history = []
    market_timestamps = []

    while not done:
        action_id = choose_best_action_with_lookahead(env, action_ids, future_window_steps)
        _, _, done, info = env.step(action_id)
        portfolio_values.append(float(info.get("portfolio_value", env.portfolio_value)))
        effective_action = int(info.get("effective_action", action_id))
        actions_history.append(effective_action)
        strategy_names_history.append(action_names.get(effective_action, f"Action {effective_action}"))
        market_timestamps.append(engine._extract_step_market_timestamp(env, info.get("step", len(actions_history))))

    initial_capital = float(env_config["initial_capital"])
    cumulative_returns = [
        (pv - initial_capital) / initial_capital if initial_capital > 0 else 0.0
        for pv in portfolio_values
    ]

    return {
        "policy_name": "future_window_best_strategy",
        "future_window_steps": int(future_window_steps),
        "actions_history": actions_history,
        "strategy_names_history": strategy_names_history,
        "market_timestamps": market_timestamps,
        "portfolio_values": portfolio_values,
        "cumulative_returns": cumulative_returns,
        "final_return": float(cumulative_returns[-1]) if cumulative_returns else 0.0,
    }


def build_performance_table(data: Dict) -> pd.DataFrame:
    initial_capital = float(data.get("initial_capital", 1_000_000))
    rows = []

    rl_metrics = curve_metrics(
        cumulative_returns=data.get("cumulative_returns", []),
        market_timestamps=data.get("market_timestamps", []),
        initial_capital=initial_capital,
        actions=data.get("actions_history", []),
    )
    rows.append({"name": "RL Agent", "category": "RL", **rl_metrics})

    for name, curve_data in data.get("benchmark_curves", {}).items():
        metrics = curve_metrics(
            cumulative_returns=curve_data.get("cumulative_returns", []),
            market_timestamps=curve_data.get("market_timestamps", data.get("market_timestamps", [])),
            initial_capital=initial_capital,
            actions=curve_data.get("actions_history", []),
        )
        rows.append({"name": name, "category": "Benchmark", **metrics})

    df = pd.DataFrame(rows)
    df = df.sort_values(["category", "sharpe_ratio", "total_return"], ascending=[True, False, False]).reset_index(drop=True)
    return df


def save_performance_table(df: pd.DataFrame, episode_dir: str) -> None:
    csv_path = os.path.join(episode_dir, "strategy_performance_table.csv")
    md_path = os.path.join(episode_dir, "strategy_performance_table.md")
    png_path = os.path.join(episode_dir, "strategy_performance_table.png")

    df.to_csv(csv_path, index=False)

    md_df = df.copy()
    for col in ["total_return", "annualized_return", "volatility", "max_drawdown", "win_rate"]:
        md_df[col] = md_df[col].map(lambda x: f"{x * 100:.2f}%")
    for col in ["sharpe_ratio", "calmar_ratio"]:
        md_df[col] = md_df[col].map(lambda x: f"{x:.3f}")
    md_df["final_portfolio_value"] = md_df["final_portfolio_value"].map(lambda x: f"{x:,.0f}")
    md_df["switch_count"] = md_df["switch_count"].map(lambda x: f"{int(x)}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_df.to_markdown(index=False))
        f.write("\n")

    fig, ax = plt.subplots(figsize=(16, max(4.5, 0.45 * (len(df) + 2))))
    ax.axis("off")
    display_df = md_df.copy()
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_returns(data: Dict, episode_dir: str) -> None:
    rl_ts = pd.to_datetime(pd.Series(data.get("market_timestamps", [])), errors="coerce")
    rl_curve = np.asarray(data.get("cumulative_returns", []), dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(rl_ts, rl_curve, label="RL Agent", linewidth=2.6, color="black")

    for name, curve_data in data.get("benchmark_curves", {}).items():
        ts = pd.to_datetime(pd.Series(curve_data.get("market_timestamps", [])), errors="coerce")
        curve = np.asarray(curve_data.get("cumulative_returns", []), dtype=float) * 100.0
        if len(ts) != len(curve):
            continue
        width = 2.0 if name == "Best Strategy" else 1.2
        alpha = 0.95 if name in {"Best Strategy", "Best Fixed Strategy", "Buy & Hold"} else 0.8
        ax.plot(ts, curve, label=name, linewidth=width, alpha=alpha)

    ax.set_title("RL vs Benchmarks Cumulative Returns")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Cumulative Return (%)")
    ax.grid(True, alpha=0.25)
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(episode_dir, "all_benchmarks_cumulative_returns.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_rl_strategy_switches(data: Dict, episode_dir: str) -> None:
    timestamps = pd.to_datetime(pd.Series(data.get("market_timestamps", [])), errors="coerce")
    actions = data.get("actions_history", [])
    strategy_names = strategy_name_map()
    if len(timestamps) == 0 or len(actions) == 0:
        return

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.step(timestamps, actions, where="post", linewidth=1.8, color="tab:blue")
    ax.scatter(timestamps, actions, s=16, color="tab:blue", alpha=0.8)
    ax.set_yticks(sorted(strategy_names.keys()))
    ax.set_yticklabels([strategy_names[i] for i in sorted(strategy_names.keys())])
    ax.set_title("RL Strategy Switching Path")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Selected Strategy")
    ax.grid(True, axis="x", alpha=0.25)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(episode_dir, "rl_strategy_switches.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def enhance_json(data: Dict, fixed_action_curves: Dict[str, Dict], oracle_curve: Dict) -> Dict:
    benchmark_curves = dict(data.get("benchmark_curves", {}))

    original_best = benchmark_curves.get("Best Strategy")
    if original_best is not None:
        benchmark_curves["Best Fixed Strategy"] = original_best

    benchmark_curves["Best Strategy"] = {
        "cumulative_returns": oracle_curve.get("cumulative_returns", []),
        "market_timestamps": oracle_curve.get("market_timestamps", []),
        "actions_history": oracle_curve.get("actions_history", []),
        "strategy_names_history": oracle_curve.get("strategy_names_history", []),
        "future_window_steps": int(oracle_curve.get("future_window_steps", 1)),
        "final_return": float(oracle_curve.get("final_return", 0.0)),
    }

    for strategy_name, curve in fixed_action_curves.items():
        benchmark_curves[strategy_name] = curve

    data["benchmark_curves"] = benchmark_curves
    data["benchmark_metadata"] = {
        "best_strategy_definition": "At each decision point, choose the action strategy with the highest realized return over the future lookahead window.",
        "future_window_steps": int(oracle_curve.get("future_window_steps", 1)),
        "action_strategy_names": list(fixed_action_curves.keys()),
    }
    return data


def save_json(json_path: str, data: Dict) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    episode_dir = os.path.join(REPO_ROOT, args.episode_dir)
    config_path = os.path.join(REPO_ROOT, args.config_path)

    json_path, backtest_data = load_backtest_json(episode_dir)
    backup_json(json_path)

    cfg = load_config(config_path)
    data_dict = prepare_data(cfg)
    test_data = data_dict["test"]

    env_config = dict(backtest_data.get("env_config", {}))
    if not env_config:
        env_config = {
            "initial_capital": cfg.get("backtest", {}).get("initial_capital", 1_000_000),
            "period_minutes": cfg.get("environment", {}).get("period_minutes", 1440),
            "state_lookback_days": cfg.get("environment", {}).get("state_lookback_days", 5),
            "switch_period_multiplier": cfg.get("environment", {}).get("switch_period_multiplier", 1),
            "transaction_cost": cfg.get("environment", {}).get("transaction_cost", 0.001),
            "dynamic_switch": cfg.get("environment", {}).get("dynamic_switch", True),
            "state_update_frequency": cfg.get("environment", {}).get("state_update_frequency", "dynamic"),
            "allow_short": cfg.get("environment", {}).get("allow_short", True),
            "max_short_per_symbol_ratio": cfg.get("environment", {}).get("max_short_per_symbol_ratio", 0.1),
            "max_total_short_ratio": cfg.get("environment", {}).get("max_total_short_ratio", 0.25),
            "max_position_ratio": cfg.get("environment", {}).get("max_position_ratio", 0.3),
            "total_position_ratio": cfg.get("environment", {}).get("total_position_ratio", 0.8),
            "rebalance_tolerance": cfg.get("environment", {}).get("rebalance_tolerance", 0.02),
            "max_trade_fraction": cfg.get("environment", {}).get("max_trade_fraction", 0.01),
            "slippage_rate": cfg.get("environment", {}).get("slippage_rate", 0.0002),
            "impact_rate": cfg.get("environment", {}).get("impact_rate", 0.0001),
            "min_commission": cfg.get("environment", {}).get("min_commission", 2.0),
            "min_signal_threshold": cfg.get("environment", {}).get("min_signal_threshold", 0.08),
            "signal_power": cfg.get("environment", {}).get("signal_power", 1.5),
            "profit_lock_enabled": cfg.get("environment", {}).get("profit_lock_enabled", False),
            "profit_lock_min_return": cfg.get("environment", {}).get("profit_lock_min_return", 0.03),
            "profit_lock_drawdown_threshold": cfg.get("environment", {}).get("profit_lock_drawdown_threshold", 0.08),
            "profit_lock_cooldown_steps": cfg.get("environment", {}).get("profit_lock_cooldown_steps", 10),
            "profit_lock_safe_action": cfg.get("environment", {}).get("profit_lock_safe_action", 0),
            "min_action_hold_steps": cfg.get("environment", {}).get("min_action_hold_steps", 1),
        }

    fixed_curves = simulate_fixed_action_benchmarks(
        test_data=test_data,
        env_config=env_config,
        data_path=cfg["data"]["data_path"],
        bar_frequency=str(cfg["data"].get("bar_frequency", "daily")).lower(),
    )
    oracle_curve = simulate_future_best_strategy(
        test_data=test_data,
        env_config=env_config,
        data_path=cfg["data"]["data_path"],
        bar_frequency=str(cfg["data"].get("bar_frequency", "daily")).lower(),
        future_window_steps=args.future_window_steps,
    )

    enhanced_data = enhance_json(backtest_data, fixed_curves, oracle_curve)
    save_json(json_path, enhanced_data)

    performance_df = build_performance_table(enhanced_data)
    save_performance_table(performance_df, episode_dir)
    plot_cumulative_returns(enhanced_data, episode_dir)
    plot_rl_strategy_switches(enhanced_data, episode_dir)

    print(f"Enhanced JSON saved to: {json_path}")
    print(f"Performance table rows: {len(performance_df)}")
    print(f"Future-window oracle steps: {args.future_window_steps}")


if __name__ == "__main__":
    main()
