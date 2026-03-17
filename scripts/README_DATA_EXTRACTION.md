# 数据提取脚本使用说明

## 功能说明

`extract_data.py` 脚本用于从 `G:\论文数据` 提取ZIP压缩的分钟级股票数据，并转换为系统需要的20分钟聚合格式。

## 数据处理流程

1. **扫描ZIP文件**：遍历 `G:\论文数据` 目录下所有ZIP压缩包
2. **解压提取**：从每个ZIP中提取CSV文件
3. **列名转换**：将中文列名转换为英文（如 `收盘价_ClPr` → `close`）
4. **时间处理**：合并日期和时间列为标准时间戳格式
5. **20分钟聚合**：
   - Open = 第一分钟的开盘价
   - High = 20分钟内最高价
   - Low = 20分钟内最低价
   - Close = 最后一分钟的收盘价
   - Volume = 20分钟累计成交量
6. **按年保存**：输出到 `trading_rl_system/data/raw/stock_data_YYYY.csv`

## 使用方法

### 方式1：交互式运行

```bash
cd trading_rl_system
python scripts/extract_data.py
```

然后根据提示选择处理模式：
- 模式1：测试模式（仅处理前5个文件，快速验证）
- 模式2：处理2024年数据
- 模式3：处理2023年数据
- 模式4：处理2022年数据
- 模式5：处理所有数据（2022-2024）

### 方式2：直接在代码中调用

```python
from scripts.extract_data import DataExtractor

# 创建提取器
extractor = DataExtractor(
    source_dir=r"G:\论文数据",
    output_dir=r"trading_rl_system\data\raw",
    resample_minutes=20
)

# 测试模式（处理前5个文件）
extractor.extract_all_data(max_files=5)

# 或处理特定年份
extractor.extract_all_data(year_filter=2024)

# 或处理所有数据
extractor.extract_all_data()
```

## 输出格式

输出的CSV文件包含以下列：

| 列名 | 说明 | 类型 |
|------|------|------|
| timestamp | 时间戳（20分钟周期） | datetime |
| symbol | 股票代码（如600000） | string |
| open | 开盘价 | float |
| high | 最高价 | float |
| low | 最低价 | float |
| close | 收盘价 | float |
| volume | 成交量 | float |
| amount | 成交额 | float |

## 注意事项

1. **处理时间**：全量数据处理可能需要数小时，建议先用测试模式验证
2. **磁盘空间**：确保有足够空间存储输出文件（每年约几GB）
3. **内存占用**：大文件处理时内存占用较高，建议16GB以上内存
4. **编码问题**：原始CSV使用GBK编码，输出使用UTF-8编码

## 故障排查

### 问题1：找不到ZIP文件
- 检查 `G:\论文数据` 路径是否正确
- 确认ZIP文件名包含 `RESSET_STKSH` 或 `RESSET_STKSZ`

### 问题2：解压失败
- 检查ZIP文件是否损坏
- 确认有足够的临时空间（脚本会创建 `temp_extract` 目录）

### 问题3：列名不匹配
- 检查CSV文件的列名是否与脚本中的 `column_mapping` 一致
- 如有差异，修改 `column_mapping` 字典

### 问题4：内存不足
- 减少 `max_files` 参数，分批处理
- 或按年份分别处理

## 后续步骤

数据提取完成后：

1. 检查输出文件：`trading_rl_system/data/raw/stock_data_*.csv`
2. 验证数据质量（时间连续性、价格合理性等）
3. 使用 `DataLoader` 加载数据进行训练

```python
from src.data.data_loader import DataLoader

loader = DataLoader('configs/data_config.yaml')
data = loader.load_data('data/raw/stock_data_2024.csv')
```
