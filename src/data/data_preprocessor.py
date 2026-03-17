"""
数据预处理器
Data Preprocessor
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    """
    数据预处理器：处理缺失值、聚合周期、标准化
    """
    
    def __init__(self, period_minutes: int = 20, bar_frequency: str = 'minute'):
        """
        初始化预处理器
        
        Args:
            period_minutes: 聚合周期（分钟）
            bar_frequency: K线频率（minute 或 daily）
        """
        self.period_minutes = period_minutes
        self.bar_frequency = str(bar_frequency).lower()
        self.price_scaler = MinMaxScaler()
        self.indicator_scaler = StandardScaler()
        self.fitted = False
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 原始数据
            method: 填充方法 ('ffill', 'bfill', 'interpolate')
        
        Returns:
            df: 处理后的数据
        """
        df = df.copy()
        
        # 按股票分组处理
        result_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            if method == 'ffill':
                # 前向填充
                symbol_df = symbol_df.ffill()
            elif method == 'bfill':
                # 后向填充
                symbol_df = symbol_df.bfill()
            elif method == 'interpolate':
                # 线性插值
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                symbol_df[numeric_columns] = symbol_df[numeric_columns].interpolate(method='linear')
            
            # 如果还有缺失值，用0填充（主要是volume）
            symbol_df = symbol_df.fillna(0)
            
            result_dfs.append(symbol_df)
        
        df = pd.concat(result_dfs, ignore_index=True)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        
        return df
    
    def aggregate_to_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将分钟数据聚合到指定周期
        
        Args:
            df: 分钟级数据
        
        Returns:
            period_df: 聚合后的数据
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 检查数据是否足够进行聚合
        if df['timestamp'].nunique() <= 1:
            print(f"警告: 数据时间戳唯一值不足，无法进行{self.period_minutes}分钟聚合")
            print(f"  时间戳唯一值: {df['timestamp'].nunique()}")
            print(f"  使用原始数据（不进行聚合）")
            return df
        
        # 按股票分组聚合
        result_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # 检查该股票的数据是否足够
            if symbol_df['timestamp'].nunique() <= 1:
                print(f"警告: 股票{symbol}数据时间戳唯一值不足，跳过聚合")
                symbol_df['period_timestamp'] = symbol_df['timestamp']
                result_dfs.append(symbol_df)
                continue
                
            symbol_df = symbol_df.set_index('timestamp')
            
            # 聚合规则
            agg_dict = {
                'open': 'first',      # 周期开盘价 = 第一分钟开盘价
                'high': 'max',        # 周期最高价 = 周期内最高价
                'low': 'min',         # 周期最低价 = 周期内最低价
                'close': 'last',      # 周期收盘价 = 最后一分钟收盘价
                'volume': 'sum'       # 周期成交量 = 周期内成交量之和
            }
            
            # 重采样
            try:
                period_df = symbol_df.resample(f'{self.period_minutes}min').agg(agg_dict)

                # 删除非交易时段的伪样本：
                # 使用sum聚合时，volume会在空窗口变成0，导致dropna(how='all')无法剔除周末/停牌空窗。
                # 因此以价格列是否存在为准过滤。
                period_df = period_df.dropna(subset=['open', 'high', 'low', 'close'])
                
                if period_df.empty:
                    print(f"警告: 股票{symbol}聚合后数据为空，使用原始数据")
                    symbol_df_reset = symbol_df.reset_index()
                    symbol_df_reset['period_timestamp'] = symbol_df_reset['timestamp']
                    result_dfs.append(symbol_df_reset)
                    continue
                
                # 添加股票代码
                period_df['symbol'] = symbol
                
                # 重置索引并重命名
                period_df = period_df.reset_index()
                period_df = period_df.rename(columns={'timestamp': 'period_timestamp'})
                
                result_dfs.append(period_df)
                
            except Exception as e:
                print(f"警告: 股票{symbol}聚合失败: {e}")
                print(f"  使用原始数据")
                symbol_df_reset = symbol_df.reset_index()
                symbol_df_reset['period_timestamp'] = symbol_df_reset['timestamp']
                result_dfs.append(symbol_df_reset)
        
        # 合并所有股票
        if not result_dfs:
            print("错误: 所有股票聚合失败，返回原始数据")
            return df
        
        result = pd.concat(result_dfs, ignore_index=True)
        
        # 重命名timestamp列为原始时间戳，使用period_timestamp作为聚合后的时间戳
        if 'period_timestamp' in result.columns:
            result = result.rename(columns={'timestamp': 'original_timestamp'})
            result = result.rename(columns={'period_timestamp': 'timestamp'})
        
        result = result.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        
        # 确保没有NaN值（按股票独立处理，避免跨股票前向填充污染）
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in result.columns:
                # 用前向填充处理NaN
                result[col] = result.groupby('symbol')[col].ffill()
                # 如果还有NaN，用0填充
                result[col] = result[col].fillna(0)
        
        print(f"聚合完成: {len(result)} 行数据")
        return result

    def aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将分钟数据聚合到交易日日线

        规则：
        - 按 symbol + trading_date 分组
        - open=first, high=max, low=min, close=last, volume=sum
        - timestamp 固定为当日 15:00，避免自然时间窗锚点偏移
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        if df.empty:
            return df

        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        df['trading_date'] = df['timestamp'].dt.date

        grouped = df.groupby(['symbol', 'trading_date'], as_index=False).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        grouped['timestamp'] = pd.to_datetime(grouped['trading_date'].astype(str) + ' 15:00:00')
        grouped['original_timestamp'] = grouped['timestamp']
        grouped = grouped.drop(columns=['trading_date'])
        grouped = grouped.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in grouped.columns:
                grouped[col] = grouped[col].replace([np.inf, -np.inf], np.nan)
                grouped[col] = grouped.groupby('symbol')[col].ffill().fillna(0.0)

        print(f"日度聚合完成: {len(grouped)} 行数据")
        return grouped
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        移除异常值
        
        Args:
            df: 数据
            method: 方法 ('iqr', 'zscore')
            threshold: 阈值
        
        Returns:
            df: 处理后的数据
        """
        df = df.copy()
        
        price_columns = ['open', 'high', 'low', 'close']
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df.loc[mask, price_columns]
            
            if method == 'iqr':
                # IQR方法
                Q1 = symbol_data.quantile(0.25)
                Q3 = symbol_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # 将异常值替换为边界值
                for col in price_columns:
                    df.loc[mask, col] = df.loc[mask, col].clip(
                        lower=lower_bound[col], 
                        upper=upper_bound[col]
                    )
            
            elif method == 'zscore':
                # Z-score方法
                mean = symbol_data.mean()
                std = symbol_data.std()
                
                for col in price_columns:
                    z_scores = np.abs((df.loc[mask, col] - mean[col]) / std[col])
                    outliers = z_scores > threshold
                    
                    if outliers.any():
                        # 用中位数替换异常值
                        median_val = df.loc[mask, col].median()
                        df.loc[mask & outliers, col] = median_val
        
        return df
    
    def normalize(self, df: pd.DataFrame, columns: Optional[list] = None, 
                  method: str = 'minmax') -> pd.DataFrame:
        """
        数据标准化
        
        Args:
            df: 数据
            columns: 需要标准化的列（None表示所有数值列）
            method: 标准化方法 ('minmax', 'standard')
        
        Returns:
            df: 标准化后的数据
        """
        df = df.copy()
        
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
        
        # 按股票分组标准化
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            df.loc[mask, columns] = scaler.fit_transform(df.loc[mask, columns])
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加时间特征
        
        Args:
            df: 数据
        
        Returns:
            df: 添加时间特征后的数据
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 提取时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # 是否是交易日开盘/收盘时段
        df['is_market_open'] = ((df['hour'] == 9) & (df['minute'] >= 30)) | \
                               ((df['hour'] >= 10) & (df['hour'] < 15))
        
        return df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算收益率并对volume和amount列进行对数变换
        
        Args:
            df: 数据
        
        Returns:
            df: 添加收益率列的数据
        """
        df = df.copy()
        
        # 对volume列进行对数变换
        if 'volume' in df.columns:
            df['volume'] = np.log(df['volume'] + 1)
        
        # 对amount列进行对数变换（如果存在）
        if 'amount' in df.columns:
            df['amount'] = np.log(df['amount'] + 1)
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            
            # 简单收益率
            df.loc[mask, 'return'] = df.loc[mask, 'close'].pct_change()
            
            # 对数收益率
            df.loc[mask, 'log_return'] = np.log(df.loc[mask, 'close'] / df.loc[mask, 'close'].shift(1))
        
        # 填充第一行的NaN
        df['return'] = df['return'].fillna(0)
        df['log_return'] = df['log_return'].fillna(0)
        
        return df
    
    def preprocess_pipeline(self, df: pd.DataFrame,
                           handle_missing: bool = True,
                           aggregate: bool = True,
                           remove_outliers: bool = True,
                           add_time_features: bool = False,
                           calculate_returns: bool = True) -> pd.DataFrame:
        """
        完整的预处理流程
        
        Args:
            df: 原始数据
            handle_missing: 是否处理缺失值
            aggregate: 是否聚合到周期
            remove_outliers: 是否移除异常值
            add_time_features: 是否添加时间特征
            calculate_returns: 是否计算收益率
        
        Returns:
            df: 预处理后的数据
        """
        print("Starting preprocessing pipeline...")
        
        if handle_missing:
            print("  - Handling missing values...")
            df = self.handle_missing_values(df)
        
        if aggregate:
            if self.bar_frequency == 'daily':
                print("  - Aggregating to daily bars...")
                df = self.aggregate_to_daily(df)
            else:
                print(f"  - Aggregating to {self.period_minutes}-minute periods...")
                df = self.aggregate_to_period(df)
        
        if remove_outliers:
            print("  - Removing outliers...")
            df = self.remove_outliers(df)
        
        if add_time_features:
            print("  - Adding time features...")
            df = self.add_time_features(df)
        
        if calculate_returns:
            print("  - Calculating returns...")
            df = self.calculate_returns(df)
        
        print("Preprocessing completed!")
        return df
