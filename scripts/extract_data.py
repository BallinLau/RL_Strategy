"""
数据提取脚本
从G:\论文数据提取ZIP压缩的分钟级股票数据，转换为系统需要的20分钟聚合格式
"""

import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataExtractor:
    """数据提取器：处理ZIP压缩的分钟级数据并转换为20分钟聚合格式"""
    
    def __init__(self, source_dir: str, output_dir: str, resample_minutes: int = 20):
        """
        初始化数据提取器
        
        Args:
            source_dir: 源数据目录 (G:\论文数据)
            output_dir: 输出目录 (trading_rl_system/data/raw)
            resample_minutes: 重采样周期（分钟），默认20分钟
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.resample_minutes = resample_minutes
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 列名映射：中文 -> 英文
        # 注意：这是分钟级快照数据，不是K线数据
        # 使用成交价作为当前价格，开盘价、最高价、最低价作为OHLC的参考
        self.column_mapping = {
            '行情日期_Qdate': 'date',
            '行情时间_QTime': 'time',
            '代码_Code': 'symbol',
            '开盘价(元)_Oppr': 'open',
            '最高价(元)_Hipr': 'high',
            '最低价(元)_Lopr': 'low',
            '成交价(元)_TPrice': 'close',  # 使用成交价作为收盘价
            '成交量(股)_TVolume': 'volume',
            '成交额(元)_TSum': 'amount',
        }
    
    def find_all_zip_files(self):
        """查找所有ZIP文件"""
        zip_files = []
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith('.zip') and ('RESSET_STKSH' in file or 'RESSET_STKSZ' in file):
                    file_path = Path(root) / file
                    # 检查文件是否真实存在且大小大于0
                    if file_path.exists() and file_path.stat().st_size > 0:
                        zip_files.append(file_path)
        
        logger.info(f"找到 {len(zip_files)} 个有效ZIP文件")
        return zip_files
    
    def extract_csv_from_zip(self, zip_path: Path, temp_dir: Path):
        """
        从ZIP文件中提取CSV
        
        Args:
            zip_path: ZIP文件路径
            temp_dir: 临时解压目录
            
        Returns:
            CSV文件路径
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 查找CSV文件
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if not csv_files:
                    logger.warning(f"ZIP文件中没有CSV: {zip_path}")
                    return None
                
                # 提取第一个CSV文件
                csv_file = csv_files[0]
                zip_ref.extract(csv_file, temp_dir)
                return temp_dir / csv_file
        
        except Exception as e:
            logger.error(f"解压失败 {zip_path}: {e}")
            return None
    
    def read_and_process_csv(self, csv_path: Path):
        """
        读取并处理CSV文件
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            处理后的DataFrame
        """
        try:
            # 读取CSV（使用GBK编码，因为是中文列名）
            df = pd.read_csv(csv_path, encoding='gbk', low_memory=False)
            
            # 检查必需列是否存在
            required_cols = list(self.column_mapping.keys())
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"缺少列: {missing_cols}")
                return None
            
            # 选择需要的列并重命名
            df = df[required_cols].copy()
            df.rename(columns=self.column_mapping, inplace=True)
            
            # 处理时间戳
            # 检查时间列是否已经包含日期（2022年数据格式）
            sample_time = str(df['time'].iloc[0])
            if len(sample_time) > 8:  # 如果时间列长度>8，说明包含日期
                # 时间列已包含完整日期时间，直接解析
                df['timestamp'] = pd.to_datetime(df['time'], errors='coerce')
            else:
                # 时间列只有时间，需要拼接日期
                df['timestamp'] = pd.to_datetime(
                    df['date'].astype(str) + ' ' + df['time'].astype(str),
                    format='%Y-%m-%d %H:%M:%S',
                    errors='coerce'
                )
            
            # 删除无效时间戳
            df = df.dropna(subset=['timestamp'])
            
            # 删除原始日期和时间列
            df = df.drop(['date', 'time'], axis=1)
            
            # 转换数值列
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 只删除关键价格列全为NaN的行（保留有价格信息的行）
            df = df.dropna(subset=['open', 'high', 'low'], how='all')
            
            # 按时间戳排序
            df = df.sort_values('timestamp')
            
            logger.info(f"成功读取 {len(df)} 条记录")
            return df
        
        except Exception as e:
            logger.error(f"读取CSV失败 {csv_path}: {e}")
            return None
    
    def resample_to_period(self, df: pd.DataFrame):
        """
        将分钟级数据重采样为指定周期（默认20分钟）
        
        Args:
            df: 原始分钟级DataFrame
            
        Returns:
            重采样后的DataFrame
        """
        try:
            # 设置时间戳为索引
            df = df.set_index('timestamp')
            
            # 按股票代码分组重采样
            resampled_list = []
            for symbol, group in df.groupby('symbol'):
                # 重采样规则
                resampled = group.resample(f'{self.resample_minutes}min').agg({
                    'open': 'first',      # 开盘价：第一个值
                    'high': 'max',        # 最高价：最大值
                    'low': 'min',         # 最低价：最小值
                    'close': 'last',      # 收盘价：最后一个值（成交价）
                    'volume': 'sum',      # 成交量：累计
                    'amount': 'sum'       # 成交额：累计
                })
                
                # 对于没有成交价的周期，使用开盘价填充收盘价
                resampled['close'] = resampled['close'].fillna(resampled['open'])
                
                # 删除全为NaN的行（没有任何数据的时间段）
                resampled = resampled.dropna(subset=['open', 'high', 'low'], how='all')
                
                # 填充剩余的NaN（用前值填充）
                resampled = resampled.ffill()
                
                # 填充volume和amount的NaN为0（没有交易的时间段）
                resampled['volume'] = resampled['volume'].fillna(0)
                resampled['amount'] = resampled['amount'].fillna(0)
                
                # 只删除价格列仍然是NaN的行
                resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'])
                
                # 添加股票代码列
                resampled['symbol'] = symbol
                
                resampled_list.append(resampled)
            
            # 合并所有股票
            if resampled_list:
                result = pd.concat(resampled_list)
                result = result.reset_index()
                logger.info(f"重采样后 {len(result)} 条记录")
                return result
            else:
                return None
        
        except Exception as e:
            logger.error(f"重采样失败: {e}")
            return None
    
    def process_single_zip(self, zip_path: Path, temp_dir: Path):
        """
        处理单个ZIP文件
        
        Args:
            zip_path: ZIP文件路径
            temp_dir: 临时目录
            
        Returns:
            处理后的DataFrame
        """
        try:
            logger.info(f"处理: {zip_path.name}")
            
            # 检查文件是否存在
            if not zip_path.exists():
                logger.warning(f"文件不存在: {zip_path}")
                return None
            
            # 提取CSV
            csv_path = self.extract_csv_from_zip(zip_path, temp_dir)
            if csv_path is None:
                return None
            
            # 读取并处理CSV
            df = self.read_and_process_csv(csv_path)
            if df is None or len(df) == 0:
                return None
            
            # 重采样为20分钟周期
            df_resampled = self.resample_to_period(df)
            
            # 删除临时CSV文件
            try:
                csv_path.unlink()
            except:
                pass
            
            return df_resampled
        
        except Exception as e:
            logger.error(f"处理文件失败 {zip_path.name}: {e}")
            return None
    
    def extract_all_data(self, year_filter=None, max_files=None):
        """
        提取所有数据
        
        Args:
            year_filter: 年份过滤（如 2024），None表示处理所有年份
            max_files: 最大处理文件数（用于测试），None表示处理所有文件
        """
        # 查找所有ZIP文件
        zip_files = self.find_all_zip_files()
        
        # 年份过滤 - 根据文件名中的年份标识（STKSH2024, STKSZ2024等）
        if year_filter:
            # 过滤包含年份标识的文件，如 STKSH2024, STKSZ2024
            year_patterns = [f'STKSH{year_filter}', f'STKSZ{year_filter}']
            zip_files = [f for f in zip_files if any(pattern in f.name for pattern in year_patterns)]
            logger.info(f"过滤后剩余 {len(zip_files)} 个文件（年份={year_filter}）")
        
        # 限制文件数量（测试用）
        if max_files:
            zip_files = zip_files[:max_files]
            logger.info(f"限制处理 {max_files} 个文件")
        
        # 创建临时目录
        temp_dir = Path('./temp_extract')
        temp_dir.mkdir(exist_ok=True)
        
        # 按年份分组处理
        year_data = {}
        
        for i, zip_path in enumerate(zip_files, 1):
            logger.info(f"进度: {i}/{len(zip_files)}")
            
            # 处理单个ZIP
            df = self.process_single_zip(zip_path, temp_dir)
            
            if df is not None and len(df) > 0:
                # 提取年份
                year = df['timestamp'].dt.year.iloc[0]
                
                if year not in year_data:
                    year_data[year] = []
                
                year_data[year].append(df)
        
        # 保存每年的数据
        for year, dfs in year_data.items():
            logger.info(f"合并 {year} 年数据...")
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # 按时间戳和股票代码排序
            combined_df = combined_df.sort_values(['timestamp', 'symbol'])
            
            # 保存到CSV
            output_file = self.output_dir / f'stock_data_{year}.csv'
            combined_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"保存: {output_file} ({len(combined_df)} 条记录)")
        
        # 清理临时目录
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
        
        logger.info("数据提取完成！")


def main():
    """主函数 - 交互式运行"""
    import sys
    
    # 配置路径
    source_dir = r"G:\论文数据"
    output_dir = r"trading_rl_system\data\raw"
    
    # 创建提取器
    extractor = DataExtractor(source_dir, output_dir, resample_minutes=20)
    
    # 选择处理模式
    print("=" * 60)
    print("数据提取脚本")
    print("=" * 60)
    print("1. 测试模式（仅处理前5个文件）")
    print("2. 处理2024年数据")
    print("3. 处理2023年数据")
    print("4. 处理2022年数据")
    print("5. 处理所有数据（2022-2024）")
    print("=" * 60)
    
    try:
        choice = input("请选择模式 (1-5): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n用户取消")
        sys.exit(0)
    
    if choice == '1':
        logger.info("测试模式：处理前5个文件")
        extractor.extract_all_data(max_files=5)
    elif choice == '2':
        logger.info("处理2024年数据")
        extractor.extract_all_data(year_filter=2024)
    elif choice == '3':
        logger.info("处理2023年数据")
        extractor.extract_all_data(year_filter=2023)
    elif choice == '4':
        logger.info("处理2022年数据")
        extractor.extract_all_data(year_filter=2022)
    elif choice == '5':
        logger.info("处理所有数据")
        extractor.extract_all_data()
    else:
        logger.error("无效选择")
        return
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
