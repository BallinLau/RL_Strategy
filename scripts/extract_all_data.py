"""
自动提取所有数据（无需交互）
处理G:\论文数据下的所有3933个ZIP文件
"""

from extract_data import DataExtractor
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    start_time = datetime.now()
    
    print("=" * 60)
    print("开始处理所有数据")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n预计处理:")
    print("- 约3933个ZIP文件")
    print("- 涵盖2022-2024年所有股票数据")
    print("- 预计耗时: 2-4小时")
    print("\n输出文件:")
    print("- data/raw/stock_data_2022.csv")
    print("- data/raw/stock_data_2023.csv")
    print("- data/raw/stock_data_2024.csv")
    print("\n" + "=" * 60)
    
    # 配置路径
    source_dir = r"G:\论文数据"
    output_dir = r"data\raw"
    
    # 创建提取器
    extractor = DataExtractor(source_dir, output_dir, resample_minutes=20)
    
    # 处理所有数据
    extractor.extract_all_data()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
