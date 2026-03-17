"""
数据加载器
Data Loader

增强版数据加载器，支持多股票CSV文件格式
基于EnhancedDataLoader，保持向后兼容性
"""

import os
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import warnings


class DataLoader:
    """
    数据加载器：从CSV文件加载OHLCV数据
    
    增强特性：
    1. 支持按年份组织的CSV文件（stock_data_YYYY.csv）
    2. 自动检测和加载所有可用数据文件
    3. 支持股票代码过滤和时间范围过滤
    4. 数据验证和质量检查
    5. 内存优化的大数据加载
    """
    
    def __init__(self, data_path: str, symbols: Optional[List[str]] = None):
        """
        初始化数据加载器
        
        Args:
            data_path: CSV文件路径或目录
            symbols: 股票代码列表，如果为None则加载所有股票
        """
        self.data_path = data_path
        self.symbols = symbols
        self.data_files = self._discover_data_files()
        
    def _discover_data_files(self) -> List[str]:
        """
        发现数据目录中的所有数据文件
        
        Returns:
            数据文件路径列表
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据目录不存在: {self.data_path}")
        
        # 查找所有CSV文件
        all_files = []
        for file in os.listdir(self.data_path):
            if file.endswith('.csv'):
                filepath = os.path.join(self.data_path, file)
                all_files.append(filepath)
        
        # 按文件名排序（通常是按年份）
        all_files.sort()
        
        print(f"发现 {len(all_files)} 个数据文件:")
        for file in all_files:
            print(f"  - {os.path.basename(file)}")
        
        return all_files
    
    def load_data(self, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        加载指定时间范围的数据
        
        Args:
            start_date: 开始日期 (格式: 'YYYY-MM-DD')
            end_date: 结束日期 (格式: 'YYYY-MM-DD')
        
        Returns:
            df: 包含所有股票数据的DataFrame
        """
        if not self.data_files:
            raise ValueError(f"在 {self.data_path} 中未找到数据文件")
        
        all_data_frames = []
        
        for filepath in self.data_files:
            try:
                df = self._load_single_file(filepath)
                if df is not None and not df.empty:
                    all_data_frames.append(df)
                    print(f"加载 {os.path.basename(filepath)}: {len(df)} 行")
            except Exception as e:
                print(f"警告: 加载文件 {os.path.basename(filepath)} 失败: {e}")
                continue
        
        if not all_data_frames:
            raise ValueError("没有成功加载任何数据文件")
        
        # 合并所有数据
        combined_df = pd.concat(all_data_frames, ignore_index=True)
        
        # 转换时间戳
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        
        # 按时间排序
        combined_df = combined_df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        
        # 过滤时间范围
        if start_date:
            start_dt = pd.to_datetime(start_date)
            combined_df = combined_df[combined_df['timestamp'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            combined_df = combined_df[combined_df['timestamp'] <= end_dt]
        
        # 过滤股票代码
        if self.symbols:
            # 将股票代码转换为字符串以便比较
            symbol_strs = [str(s) for s in self.symbols]
            combined_df = combined_df[combined_df['symbol'].astype(str).isin(symbol_strs)]
        
        # 验证数据
        self.validate_data(combined_df)
        
        # 数据统计
        self._print_data_statistics(combined_df)
        
        return combined_df
    
    def _load_single_file(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        加载单个数据文件
        
        Args:
            filepath: 文件路径
        
        Returns:
            df: 数据DataFrame
        """
        try:
            # 先读取表头，识别数据格式并按需读取列
            header_df = pd.read_csv(filepath, nrows=0)
            normalized_to_original = {
                self._normalize_col_name(col): col for col in header_df.columns
            }
            normalized_cols = set(normalized_to_original.keys())

            standard_required = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
            csmar_required = {'stkcd', 'trddt', 'opnprc', 'hiprc', 'loprc', 'clsprc', 'dnshrtrd'}
            crsp_required = {'permno', 'date', 'open', 'high', 'low', 'close', 'volume'}

            use_standard_schema = standard_required.issubset(normalized_cols)
            use_csmar_schema = csmar_required.issubset(normalized_cols)
            use_crsp_schema = crsp_required.issubset(normalized_cols)

            if not use_standard_schema and not use_csmar_schema and not use_crsp_schema:
                print(f"警告: 文件 {os.path.basename(filepath)} 既非标准OHLCV/TRD_Dalyr/CRSP格式，跳过")
                return None

            if use_standard_schema:
                usecols_normalized = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            elif use_csmar_schema:
                usecols_normalized = ['stkcd', 'trddt', 'opnprc', 'hiprc', 'loprc', 'clsprc', 'dnshrtrd']
                if 'trdsta' in normalized_cols:
                    usecols_normalized.append('trdsta')
                print(f"检测到TRD_Dalyr格式: {os.path.basename(filepath)}")
            else:
                usecols_normalized = ['permno', 'date', 'open', 'high', 'low', 'close', 'volume']
                print(f"检测到CRSP OHLC格式: {os.path.basename(filepath)}")

            usecols_original = [normalized_to_original[col] for col in usecols_normalized]
            dtype_map = {}
            if use_standard_schema:
                dtype_map[normalized_to_original['symbol']] = str
            elif use_csmar_schema:
                dtype_map[normalized_to_original['stkcd']] = str
                if 'trdsta' in normalized_cols:
                    dtype_map[normalized_to_original['trdsta']] = str
            else:
                dtype_map[normalized_to_original['permno']] = str

            # 使用低内存模式读取大文件
            chunksize = 100000
            chunks = []
            symbol_filter_set = set(str(s) for s in self.symbols) if self.symbols else None

            for chunk in pd.read_csv(
                filepath,
                usecols=usecols_original,
                dtype=dtype_map,
                chunksize=chunksize
            ):
                chunk.columns = [self._normalize_col_name(col) for col in chunk.columns]

                if use_csmar_schema:
                    # 仅保留正常交易日，避免停牌/ST状态影响训练稳定性
                    if 'trdsta' in chunk.columns:
                        chunk = chunk[chunk['trdsta'].astype(str) == '1']
                    if chunk.empty:
                        continue
                    chunk = chunk.rename(columns={
                        'stkcd': 'symbol',
                        'trddt': 'timestamp',
                        'opnprc': 'open',
                        'hiprc': 'high',
                        'loprc': 'low',
                        'clsprc': 'close',
                        'dnshrtrd': 'volume'
                    })
                elif use_crsp_schema:
                    chunk = chunk.rename(columns={
                        'permno': 'symbol',
                        'date': 'timestamp'
                    })

                chunk = chunk[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                chunk['symbol'] = chunk['symbol'].astype(str)

                if symbol_filter_set is not None:
                    chunk = chunk[chunk['symbol'].isin(symbol_filter_set)]
                    if chunk.empty:
                        continue

                chunks.append(chunk)
            
            if not chunks:
                return None
            
            df = pd.concat(chunks, ignore_index=True)
            
            # 标准化列名
            df.columns = [self._normalize_col_name(col) for col in df.columns]
            
            # 确保必需的列存在
            required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"警告: 文件 {os.path.basename(filepath)} 缺少列: {missing_columns}")
                return None
            
            # 选择必需的列
            df = df[required_columns].copy()
            
            # 清理数据
            df = self._clean_data(df)
            
            return df
            
        except Exception as e:
            print(f"加载文件 {os.path.basename(filepath)} 时出错: {e}")
            return None

    @staticmethod
    def _normalize_col_name(col_name: str) -> str:
        """
        统一列名格式，兼容BOM和大小写差异
        """
        return str(col_name).replace('\ufeff', '').strip().strip('"').lower()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理数据
        
        Args:
            df: 原始数据DataFrame
        
        Returns:
            清理后的DataFrame
        """
        # 移除重复行
        df = df.drop_duplicates()
        
        # 处理缺失值
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                # 使用前向填充处理缺失值
                df[col] = df[col].ffill()
        
        # 移除仍有缺失值的行
        df = df.dropna(subset=numeric_cols)
        
        # 确保价格数据合理
        df = df[
            (df['open'] > 0) &
            (df['high'] > 0) &
            (df['low'] > 0) &
            (df['close'] > 0) &
            (df['volume'] >= 0) &
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        ]
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据完整性
        
        Args:
            df: 数据DataFrame
        
        Returns:
            valid: 数据是否有效
        """
        if df.empty:
            print("错误: DataFrame为空")
            return False
        
        # 检查必需的列
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"错误: 缺少列: {missing_columns}")
            return False
        
        # 检查缺失值
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            warnings.warn(f"发现缺失值:\n{null_counts[null_counts > 0]}")
        
        # 检查价格逻辑
        invalid_prices = df[
            (df['high'] < df['low']) | 
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) |
            (df['low'] > df['open']) | 
            (df['low'] > df['close'])
        ]
        
        if len(invalid_prices) > 0:
            print(f"警告: 发现 {len(invalid_prices)} 行价格关系不合理")
        
        # 检查负值
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        negative_values = (df[price_columns] < 0).any()
        
        if negative_values.any():
            print(f"警告: 发现负值列: {negative_values[negative_values].index.tolist()}")
        
        return True
    
    def _print_data_statistics(self, df: pd.DataFrame):
        """
        打印数据统计信息
        
        Args:
            df: 数据DataFrame
        """
        print("\n" + "="*80)
        print("数据加载统计")
        print("="*80)
        
        print(f"总行数: {len(df):,}")
        print(f"总列数: {len(df.columns)}")
        
        if not df.empty:
            print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
            print(f"总天数: {(df['timestamp'].max() - df['timestamp'].min()).days} 天")
            
            # 股票统计
            unique_symbols = df['symbol'].unique()
            print(f"唯一股票代码数: {len(unique_symbols)}")
            
            # 显示前10个股票代码
            if len(unique_symbols) <= 10:
                print(f"股票代码: {', '.join(map(str, unique_symbols))}")
            else:
                print(f"前10个股票代码: {', '.join(map(str, unique_symbols[:10]))}")
            
            # 时间频率分析
            time_diff = df['timestamp'].diff().dropna()
            if len(time_diff) > 0:
                mode_diff = time_diff.mode()[0]
                print(f"主要时间间隔: {mode_diff}")
            
            # 内存使用
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            print(f"内存使用: {memory_mb:.2f} MB")
        
        print("="*80)
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7, 
                   val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分训练集、验证集、测试集
        
        Args:
            df: 完整数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        
        Returns:
            train_df, val_df, test_df: 训练集、验证集、测试集
        """
        if df.empty:
            raise ValueError("数据为空，无法划分")
        
        # 按时间划分
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 获取唯一时间戳
        unique_timestamps = df['timestamp'].unique()
        n_timestamps = len(unique_timestamps)
        
        # 计算分割点
        train_end = int(n_timestamps * train_ratio)
        val_end = int(n_timestamps * (train_ratio + val_ratio))
        
        # 分割时间戳
        train_timestamps = unique_timestamps[:train_end]
        val_timestamps = unique_timestamps[train_end:val_end]
        test_timestamps = unique_timestamps[val_end:]
        
        # 分割数据
        train_df = df[df['timestamp'].isin(train_timestamps)].reset_index(drop=True)
        val_df = df[df['timestamp'].isin(val_timestamps)].reset_index(drop=True)
        test_df = df[df['timestamp'].isin(test_timestamps)].reset_index(drop=True)
        
        print(f"数据划分:")
        print(f"  训练集: {len(train_df)} 行 ({len(train_timestamps)} 个时间点)")
        print(f"  验证集: {len(val_df)} 行 ({len(val_timestamps)} 个时间点)")
        print(f"  测试集: {len(test_df)} 行 ({len(test_timestamps)} 个时间点)")
        
        return train_df, val_df, test_df
    
    def get_date_range(self, df: pd.DataFrame) -> Tuple[str, str]:
        """
        获取数据的日期范围
        
        Args:
            df: 数据DataFrame
        
        Returns:
            start_date, end_date: 开始和结束日期
        """
        if df.empty:
            return "未知", "未知"
        
        start_date = df['timestamp'].min().strftime('%Y-%m-%d')
        end_date = df['timestamp'].max().strftime('%Y-%m-%d')
        return start_date, end_date
    
    def get_available_symbols(self) -> List[str]:
        """
        获取数据中可用的股票代码
        
        Returns:
            股票代码列表
        """
        if not self.data_files:
            return []
        
        all_symbols = set()
        
        for filepath in self.data_files[:3]:  # 只检查前3个文件以加快速度
            try:
                # 读取文件的前1000行来获取股票代码
                df_sample = pd.read_csv(filepath, nrows=1000)
                if 'symbol' in df_sample.columns:
                    symbols = df_sample['symbol'].unique()
                    all_symbols.update(map(str, symbols))
            except:
                continue
        
        return sorted(list(all_symbols))
