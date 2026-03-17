"""
数据筛选脚本
根据Stkcd.csv中的100家公司股票编号，从已提炼的数据中筛选出对应公司的数据
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFilter:
    """数据筛选器：根据股票列表筛选数据"""
    
    def __init__(self, stock_list_file: str, input_dir: str, output_dir: str):
        """
        初始化数据筛选器
        
        Args:
            stock_list_file: 股票列表文件路径 (Stkcd.csv)
            input_dir: 输入数据目录 (data/raw)
            output_dir: 输出目录 (data/filtered)
        """
        self.stock_list_file = Path(stock_list_file)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取股票列表
        self.stock_codes = self._load_stock_list()
        logger.info(f"加载了 {len(self.stock_codes)} 个股票代码")
    
    def _load_stock_list(self):
        """读取股票列表"""
        try:
            df = pd.read_csv(self.stock_list_file)
            # 获取股票代码列（假设列名为 'Stkcd'）
            stock_codes = df['Stkcd'].astype(str).str.strip().tolist()
            
            # 生成所有可能的格式
            all_formats = set()
            for code in stock_codes:
                # 原始6位代码
                all_formats.add(code)
                # 带后缀格式
                all_formats.add(f"{code}.SZ")
                all_formats.add(f"{code}.SH")
                # 转换为整数格式（去掉前导0）
                try:
                    code_int = int(code)
                    all_formats.add(code_int)
                    all_formats.add(str(code_int))
                except:
                    pass
            
            return all_formats
        except Exception as e:
            logger.error(f"读取股票列表失败: {e}")
            raise
    
    def filter_single_file(self, input_file: Path, output_file: Path, chunksize: int = 100000):
        """
        筛选单个数据文件
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            chunksize: 分块读取大小（行数）
        """
        try:
            logger.info(f"开始处理: {input_file.name}")
            start_time = datetime.now()
            
            # 分块读取并筛选
            filtered_chunks = []
            total_rows = 0
            filtered_rows = 0
            
            # 使用chunksize分块读取大文件
            for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize), 1):
                total_rows += len(chunk)
                
                # 筛选：保留stock_codes中的股票
                # 处理symbol列可能的格式：纯代码或带后缀
                filtered_chunk = chunk[chunk['symbol'].isin(self.stock_codes)]
                
                if len(filtered_chunk) > 0:
                    filtered_chunks.append(filtered_chunk)
                    filtered_rows += len(filtered_chunk)
                
                if i % 10 == 0:
                    logger.info(f"  已处理 {total_rows:,} 行，筛选出 {filtered_rows:,} 行")
            
            # 合并所有筛选结果
            if filtered_chunks:
                result_df = pd.concat(filtered_chunks, ignore_index=True)
                
                # 保存到输出文件
                result_df.to_csv(output_file, index=False, encoding='utf-8')
                
                end_time = datetime.now()
                duration = end_time - start_time
                
                logger.info(f"完成: {output_file.name}")
                logger.info(f"  原始数据: {total_rows:,} 行")
                logger.info(f"  筛选后: {filtered_rows:,} 行 ({filtered_rows/total_rows*100:.2f}%)")
                logger.info(f"  耗时: {duration}")
                logger.info(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
                
                return filtered_rows
            else:
                logger.warning(f"未找到匹配的股票数据: {input_file.name}")
                return 0
        
        except Exception as e:
            logger.error(f"处理文件失败 {input_file.name}: {e}")
            raise
    
    def filter_all_years(self, years=[2022, 2023, 2024, 2025]):
        """
        筛选所有年份的数据
        
        Args:
            years: 要处理的年份列表
        """
        print("=" * 60)
        print("数据筛选脚本")
        print("=" * 60)
        print(f"股票列表: {self.stock_list_file}")
        print(f"股票数量: {len([c for c in self.stock_codes if isinstance(c, str) and '.' not in c and len(c) == 6])} 个")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)
        
        total_filtered = 0
        
        for year in years:
            input_file = self.input_dir / f"stock_data_{year}.csv"
            output_file = self.output_dir / f"stock_data_{year}_filtered.csv"
            
            if not input_file.exists():
                logger.warning(f"文件不存在: {input_file}")
                continue
            
            print(f"\n处理 {year} 年数据...")
            filtered_rows = self.filter_single_file(input_file, output_file)
            total_filtered += filtered_rows
        
        print("\n" + "=" * 60)
        print("筛选完成！")
        print(f"总共筛选出: {total_filtered:,} 行数据")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)


def main():
    """主函数"""
    # 配置路径
    stock_list_file = "Stkcd.csv"
    input_dir = "data/raw"
    output_dir = "data/filtered"
    
    # 创建筛选器
    filter_tool = DataFilter(stock_list_file, input_dir, output_dir)
    
    # 筛选所有年份数据
    filter_tool.filter_all_years(years=[2022, 2023, 2024, 2025])


if __name__ == '__main__':
    main()
