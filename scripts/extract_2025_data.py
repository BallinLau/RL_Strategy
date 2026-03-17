"""
提取2025年数据脚本
从G:\论文数据\2025文件夹提取ZIP压缩的分钟级股票数据，转换为20分钟聚合格式
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


class Data2025Extractor:
    """2025年数据提取器：处理ZIP压缩的分钟级数据并转换为20分钟聚合格式"""
    
    def __init__(self, source_dir: str, output_dir: str, resample_minutes: int = 20):
        """
        初始化数据提取器
        
        Args:
            source_dir: 源数据目录 (G:\论文数据\2025)
            output_dir: 输出目录 (data/raw)
            resample_minutes: 重采样周期（分钟），默认20分钟
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.resample_minutes = resample_minutes
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 列名映射：2025年数据使用英文列名
        self.column_mapping = {
            'Qdate': 'date',
            'QTime': 'time',
            'Code': 'symbol',
            'Oppr': 'open',
            'Hipr': 'high',
            'Lopr': 'low',
            'TPrice': 'close',
            'TVolume': 'volume',
            'TSum': 'amount',
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
            # 尝试多种编码方式读取CSV
            encodings = ['utf-8', 'gbk', 'gb18030', 'latin1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding, low_memory=False, on_bad_lines='skip')
                    break
                except (UnicodeDecodeError, Exception) as e:
                    if encoding == encodings[-1]:  # 最后一个编码也失败
                        logger.error(f"所有编码尝试失败: {e}")
                        return None
                    continue
            
            if df is None:
                return None
            
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
            sample_time = str(df['time'].iloc[0])
            if len(sample_time) > 8:  # 如果时间列长度>8，说明包含日期
                df['timestamp'] = pd.to_datetime(df['time'], errors='coerce')
            else:
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
            
            # 只删除关键价格列全为NaN的行
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
                    'close': 'last',      # 收盘价：最后一个值
                    'volume': 'sum',      # 成交量：累计
                    'amount': 'sum'       # 成交额：累计
                })
                
                # 对于没有成交价的周期，使用开盘价填充收盘价
                resampled['close'] = resampled['close'].fillna(resampled['open'])
                
                # 删除全为NaN的行
                resampled = resampled.dropna(subset=['open', 'high', 'low'], how='all')
                
                # 填充剩余的NaN
                resampled = resampled.ffill()
                
                # 填充volume和amount的NaN为0
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
    
    def extract_2025_data(self):
        """提取2025年数据"""
        print("=" * 60)
        print("提取2025年数据")
        print("=" * 60)
        print(f"源目录: {self.source_dir}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)
        
        # 查找所有ZIP文件
        zip_files = self.find_all_zip_files()
        
        if not zip_files:
            logger.error("未找到任何ZIP文件")
            return
        
        # 创建临时目录
        temp_dir = Path('./temp_extract_2025')
        temp_dir.mkdir(exist_ok=True)
        
        # 处理所有ZIP文件
        all_data = []
        
        for i, zip_path in enumerate(zip_files, 1):
            logger.info(f"进度: {i}/{len(zip_files)}")
            
            # 处理单个ZIP
            df = self.process_single_zip(zip_path, temp_dir)
            
            if df is not None and len(df) > 0:
                all_data.append(df)
        
        # 合并所有数据
        if all_data:
            logger.info("合并2025年数据...")
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # 按时间戳和股票代码排序
            combined_df = combined_df.sort_values(['timestamp', 'symbol'])
            
            # 保存到CSV
            output_file = self.output_dir / 'stock_data_2025.csv'
            combined_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"保存: {output_file} ({len(combined_df)} 条记录)")
            logger.info(f"文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        else:
            logger.warning("没有提取到任何数据")
        
        # 清理临时目录
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
        
        print("\n" + "=" * 60)
        print("2025年数据提取完成！")
        print("=" * 60)


def main():
    """主函数"""
    # 配置路径
    source_dir = r"G:\论文数据\2025"
    output_dir = r"data\raw"
    
    # 创建提取器
    extractor = Data2025Extractor(source_dir, output_dir, resample_minutes=20)
    
    # 提取2025年数据
    extractor.extract_2025_data()


if __name__ == '__main__':
    main()
