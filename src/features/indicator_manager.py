"""
指标管理器
Indicator Manager
"""

import pandas as pd
import numpy as np
import pickle
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from .indicators import IndicatorCalculator


class IndicatorManager:
    """
    指标管理器：管理技术指标的更新和缓存
    """
    
    def __init__(self, period_minutes: int = 20, cache_dir: str = "cache/indicators"):
        """
        初始化指标管理器
        
        Args:
            period_minutes: 更新周期（分钟）
            cache_dir: 缓存目录
        """
        self.period_minutes = period_minutes
        self.last_update_time = None
        self.indicators_cache: Dict[str, pd.Series] = {}
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def should_update(self, current_time: datetime) -> bool:
        """
        判断是否需要更新指标
        
        Args:
            current_time: 当前时间
        
        Returns:
            should_update: 是否需要更新
        """
        if self.last_update_time is None:
            return True
        
        time_diff = (current_time - self.last_update_time).total_seconds() / 60
        return time_diff >= self.period_minutes
    
    def update_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        更新所有技术指标
        
        Args:
            df: 价格数据
        
        Returns:
            indicators: 所有指标的字典
        """
        calculator = IndicatorCalculator(df)
        self.indicators_cache = calculator.calculate_all()
        
        if len(df) > 0:
            self.last_update_time = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None
        
        return self.indicators_cache
    
    def _get_cache_key(self, df: pd.DataFrame) -> str:
        """
        生成数据的缓存键（基于数据内容的哈希）
        
        Args:
            df: 数据DataFrame
        
        Returns:
            cache_key: 缓存键
        """
        # 使用数据的形状、时间范围和部分内容生成哈希
        key_str = f"{df.shape}_{df['timestamp'].min()}_{df['timestamp'].max()}"
        if 'symbol' in df.columns:
            key_str += f"_{sorted(df['symbol'].unique().tolist())}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"indicators_{cache_key}.pkl")
    
    def _save_cache(self, cache_key: str, precomputed: Dict[int, Dict[str, float]]):
        """保存预计算指标到缓存"""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(precomputed, f)
            print(f"预计算指标已保存到缓存: {cache_path}")
        except Exception as e:
            print(f"警告: 保存缓存失败: {e}")
    
    def _load_cache(self, cache_key: str) -> Optional[Dict[int, Dict[str, float]]]:
        """从缓存加载预计算指标"""
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    precomputed = pickle.load(f)
                print(f"从缓存加载预计算指标: {cache_path}")
                return precomputed
            except Exception as e:
                print(f"警告: 加载缓存失败: {e}")
                return None
        return None
    
    def precompute_all_indicators(self, df: pd.DataFrame, use_cache: bool = True) -> Dict[int, Dict[str, float]]:
        """
        预计算所有时间点的技术指标（性能优化）
        
        为每个时间点预先计算指标，避免在训练过程中重复计算。
        使用增量计算优化性能。
        
        Args:
            df: 完整的历史数据
            use_cache: 是否使用缓存
        
        Returns:
            precomputed: {step: {indicator_name: value}}
        """
        # 尝试从缓存加载
        if use_cache:
            cache_key = self._get_cache_key(df)
            cached_result = self._load_cache(cache_key)
            if cached_result is not None:
                print(f"✓ 成功从缓存加载 {len(cached_result)} 个时间点的预计算指标")
                
                # 清理缓存中的nan/inf值
                print("  正在清理缓存中的nan/inf值...")
                nan_count = 0
                inf_count = 0
                for step, indicators in cached_result.items():
                    for name, value in indicators.items():
                        if pd.isna(value):
                            cached_result[step][name] = 0.0
                            nan_count += 1
                        elif np.isinf(value):
                            cached_result[step][name] = 0.0
                            inf_count += 1
                
                if nan_count > 0 or inf_count > 0:
                    print(f"  清理完成：{nan_count} 个NaN值，{inf_count} 个Inf值已替换为0")
                else:
                    print(f"  缓存数据正常，无需清理")
                
                return cached_result
        
        if df.empty:
            return {}

        # 性能优化：只在“主股票”的完整时间序列上一次性向量化计算指标，
        # 然后按时间步提取对应值，复杂度从O(N^2)降为O(N)。
        working_df = df.sort_values('timestamp').reset_index(drop=True)
        if 'symbol' in working_df.columns and working_df['symbol'].nunique() > 0:
            anchor_symbol = sorted(working_df['symbol'].astype(str).unique().tolist())[0]
            anchor_df = (
                working_df[working_df['symbol'].astype(str) == anchor_symbol]
                .sort_values('timestamp')
                .reset_index(drop=True)
            )
            print(
                f"开始预计算技术指标（向量化），主股票: {anchor_symbol}，"
                f"时间点: {len(anchor_df)}"
            )
        else:
            anchor_symbol = None
            anchor_df = working_df
            print(f"开始预计算技术指标（向量化），时间点: {len(anchor_df)}")

        calculator = IndicatorCalculator(anchor_df)
        indicators = calculator.calculate_all()

        precomputed: Dict[int, Dict[str, float]] = {}
        for step in range(len(anchor_df)):
            latest_indicators: Dict[str, float] = {}
            for name, series in indicators.items():
                if step < len(series):
                    value = series.iloc[step]
                    if pd.isna(value) or np.isinf(value):
                        latest_indicators[name] = 0.0
                    else:
                        latest_indicators[name] = float(value)
                else:
                    latest_indicators[name] = 0.0
            precomputed[step] = latest_indicators

        print(
            f"预计算完成！主股票: {anchor_symbol or 'N/A'}，"
            f"共 {len(precomputed)} 个时间点"
        )
        
        # 保存到缓存
        if use_cache:
            cache_key = self._get_cache_key(df)
            self._save_cache(cache_key, precomputed)
        
        return precomputed
    
    def get_latest_indicators(self) -> Dict[str, float]:
        """
        获取最新的指标值（用于状态空间）
        
        Returns:
            latest_indicators: 最新指标值字典
        """
        latest = {}
        for name, series in self.indicators_cache.items():
            if len(series) > 0 and not pd.isna(series.iloc[-1]):
                latest[name] = float(series.iloc[-1])
            else:
                latest[name] = 0.0
        return latest
    
    def get_indicator_at_index(self, index: int) -> Dict[str, float]:
        """
        获取指定索引位置的指标值
        
        Args:
            index: 索引位置
        
        Returns:
            indicators: 指标值字典
        """
        indicators = {}
        for name, series in self.indicators_cache.items():
            if index < len(series) and not pd.isna(series.iloc[index]):
                indicators[name] = float(series.iloc[index])
            else:
                indicators[name] = 0.0
        return indicators
    
    def get_indicator_series(self, name: str) -> Optional[pd.Series]:
        """
        获取指定指标的完整序列
        
        Args:
            name: 指标名称
        
        Returns:
            series: 指标序列
        """
        return self.indicators_cache.get(name)
    
    def clear_cache(self):
        """清空缓存"""
        self.indicators_cache.clear()
        self.last_update_time = None
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"IndicatorManager(indicators={len(self.indicators_cache)}, last_update={self.last_update_time})"
