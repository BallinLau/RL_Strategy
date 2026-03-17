"""
数据处理总脚本
自动执行两步操作：
1. 提取2025年原始数据（第一步精简：ZIP -> 20分钟聚合CSV）
2. 筛选100家公司数据（第二步缩减：全部股票 -> 100家公司）
"""

import sys
import os
from datetime import datetime

# 添加scripts目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from extract_2025_data import Data2025Extractor
from filter_data import DataFilter


def main():
    """主函数：执行完整的数据处理流程"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("数据处理总流程")
    print("=" * 80)
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n流程说明:")
    print("步骤1: 提取2025年原始数据（ZIP -> 20分钟聚合CSV）")
    print("步骤2: 筛选所有年份的100家公司数据")
    print("=" * 80)
    
    # ========== 步骤1: 提取2025年数据 ==========
    print("\n" + "=" * 80)
    print("步骤1: 提取2025年原始数据")
    print("=" * 80)
    
    try:
        source_dir = r"G:\论文数据\2025"
        output_dir = r"data\raw"
        
        extractor = Data2025Extractor(source_dir, output_dir, resample_minutes=20)
        extractor.extract_2025_data()
        
        print("\n✓ 步骤1完成：2025年数据已提取")
    except Exception as e:
        print(f"\n✗ 步骤1失败: {e}")
        print("请检查:")
        print("1. G:\\论文数据\\2025 目录是否存在")
        print("2. 目录中是否有ZIP文件")
        print("3. 是否有足够的磁盘空间")
        return
    
    # ========== 步骤2: 筛选100家公司数据 ==========
    print("\n" + "=" * 80)
    print("步骤2: 筛选100家公司数据")
    print("=" * 80)
    
    try:
        stock_list_file = "Stkcd.csv"
        input_dir = "data/raw"
        output_dir = "data/filtered"
        
        filter_tool = DataFilter(stock_list_file, input_dir, output_dir)
        filter_tool.filter_all_years(years=[2022, 2023, 2024, 2025])
        
        print("\n✓ 步骤2完成：100家公司数据已筛选")
    except Exception as e:
        print(f"\n✗ 步骤2失败: {e}")
        print("请检查:")
        print("1. Stkcd.csv 文件是否存在")
        print("2. data/raw/ 目录中是否有数据文件")
        return
    
    # ========== 完成 ==========
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("数据处理完成！")
    print("=" * 80)
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration}")
    print("\n输出文件:")
    print("- data/raw/stock_data_2025.csv (2025年全部股票)")
    print("- data/filtered/stock_data_2022_filtered.csv (2022年100家公司)")
    print("- data/filtered/stock_data_2023_filtered.csv (2023年100家公司)")
    print("- data/filtered/stock_data_2024_filtered.csv (2024年100家公司)")
    print("- data/filtered/stock_data_2025_filtered.csv (2025年100家公司)")
    print("=" * 80)


if __name__ == '__main__':
    main()
