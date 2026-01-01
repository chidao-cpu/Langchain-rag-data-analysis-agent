# 数据预处理API - get_basic_stats

## 1. API概述

数据预处理API（`get_basic_stats`）是数据分析流程中的第一个关键步骤，负责对原始数据进行全面的初始检查和基本处理。该API旨在提供数据集的完整概览，包括描述性统计信息、缺失值分析、异常值检测以及数据类型验证等功能。通过这个API，用户可以快速了解数据的质量和结构，为后续的特征工程和建模提供坚实的基础。

## 2. 功能特性

### 2.1 描述性统计分析

该API提供了全面的描述性统计功能，能够自动计算数据集的各种统计指标：

- **集中趋势度量**：均值（mean）、中位数（median）、众数（mode）
- **离散程度度量**：标准差（standard deviation）、方差（variance）、四分位距（IQR）
- **分布形态**：偏度（skewness）、峰度（kurtosis）
- **范围分析**：最小值（min）、最大值（max）、极差（range）

这些统计指标会针对不同数据类型自动调整，确保对数值型、类别型和时间序列数据都能提供有意义的分析结果。

### 2.2 缺失值分析

缺失值是数据分析中常见的问题，该API提供了全面的缺失值检测和分析功能：

- **缺失值统计**：统计每个列的缺失值数量和百分比
- **缺失值模式分析**：识别数据中缺失值的模式和相关性
- **缺失值可视化**：生成缺失值热力图，直观展示缺失值分布
- **缺失值原因推断**：基于数据上下文提供缺失值可能原因的初步推断

### 2.3 异常值检测

异常值可能会严重影响分析结果的准确性，该API集成了多种异常值检测方法：

- **统计方法**：Z-score、IQR方法（Tukey's fences）
- **机器学习方法**：隔离森林（Isolation Forest）、LOF（Local Outlier Factor）
- **可视化方法**：箱线图（box plot）、散点图矩阵
- **时间序列异常检测**：针对时间序列数据的专门方法

### 2.4 数据类型验证与转换

数据类型的正确性对于后续分析至关重要，该API提供了以下功能：

- **自动数据类型检测**：智能识别数值型、类别型、时间序列等数据类型
- **数据类型转换**：支持将数据转换为适当的类型（如字符串转日期、整数转浮点数等）
- **数据类型一致性检查**：检测和报告数据类型不一致的问题

### 2.5 数据质量评估

该API还提供了综合的数据质量评估报告：

- **完整性得分**：基于缺失值情况计算数据完整性
- **一致性得分**：评估数据的一致性和合理性
- **有效性得分**：检查数据是否符合预期格式和范围
- **及时性得分**：针对时间序列数据评估其及时性

## 3. API参数

### 3.1 基本参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `data_file` | string | None | 输入数据文件路径，支持CSV、Excel、JSON等格式 |
| `data` | DataFrame | None | 直接输入的Pandas DataFrame对象 |
| `target_columns` | list | None | 目标列列表，用于重点分析 |
| `ignore_columns` | list | None | 忽略的列列表 |

### 3.2 高级参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `detect_outliers` | bool | True | 是否执行异常值检测 |
| `outlier_methods` | list | ['iqr', 'zscore'] | 异常值检测方法列表 |
| `outlier_threshold` | float | 3.0 | 异常值检测的阈值（用于Z-score方法） |
| `analyze_missing` | bool | True | 是否执行缺失值分析 |
| `missing_threshold` | float | 0.3 | 缺失值百分比阈值，超过该阈值的列会被标记 |
| `include_visualizations` | bool | True | 是否包含可视化结果 |
| `output_format` | string | 'dict' | 输出格式，支持'dict'、'json'、'html' |

## 4. 返回值

### 4.1 基本统计信息

```python
{
    'descriptive_stats': {
        'numeric_columns': {
            'column1': {
                'count': 1000,
                'mean': 50.5,
                'median': 50.0,
                'std': 10.2,
                'min': 0,
                'max': 100,
                'skewness': 0.1,
                'kurtosis': -0.2
            },
            # 更多数值型列...
        },
        'categorical_columns': {
            'column2': {
                'count': 1000,
                'unique': 5,
                'top': 'A',
                'freq': 250
            },
            # 更多类别型列...
        }
    }
}
```

### 4.2 缺失值分析结果

```python
{
    'missing_values': {
        'total_missing': 50,
        'missing_percentage': 5.0,
        'columns_with_missing': {
            'column3': {
                'missing_count': 30,
                'missing_percentage': 3.0
            },
            # 更多包含缺失值的列...
        },
        'missing_patterns': [
            # 缺失值模式分析结果...
        ]
    }
}
```

### 4.3 异常值检测结果

```python
{
    'outliers': {
        'column1': {
            'method': 'iqr',
            'outlier_count': 25,
            'outlier_indices': [10, 25, 45, ...],
            'outlier_values': [120, 130, 115, ...]
        },
        # 更多列的异常值...
    }
}
```

### 4.4 数据质量报告

```python
{
    'data_quality': {
        'completeness_score': 0.95,
        'consistency_score': 0.98,
        'validity_score': 0.97,
        'timeliness_score': 0.90,
        'overall_score': 0.95
    }
}
```

## 5. 使用示例

### 5.1 基本用法

```python
from src.langchain_agent import LangChainDataAnalysisAgent

# 创建代理实例
agent = LangChainDataAnalysisAgent()

# 调用API获取基本统计信息
result = agent.get_basic_stats(data_file='data/sales_data.csv')

# 打印结果
print("数据集形状:", result['dataset_info']['shape'])
print("数值型列:", result['dataset_info']['numeric_cols'])
print("缺失值分析:", result['missing_values'])
```

### 5.2 高级用法

```python
# 使用DataFrame作为输入
import pandas as pd

data = pd.read_csv('data/sales_data.csv')

# 调用API并指定参数
result = agent.get_basic_stats(
    data=data,
    target_columns=['销售额', '利润'],
    ignore_columns=['订单ID'],
    detect_outliers=True,
    outlier_methods=['iqr', 'isolation_forest'],
    include_visualizations=True,
    output_format='html'
)

# 保存HTML报告
with open('data_quality_report.html', 'w') as f:
    f.write(result['html_report'])
```

## 6. 最佳实践

### 6.1 输入数据准备

- 确保输入数据格式正确，支持CSV、Excel、JSON等常见格式
- 对于大型数据集，建议先进行采样以提高处理速度
- 检查数据中是否包含不需要的列（如索引列）

### 6.2 参数选择

- 根据数据类型选择合适的异常值检测方法
- 对于时间序列数据，建议启用专门的时间序列异常检测
- 根据分析需求调整缺失值和异常值的阈值

### 6.3 结果解读

- 关注数据质量报告的综合得分
- 重点分析缺失值和异常值较多的列
- 结合业务知识判断检测到的异常值是否真的是异常

### 6.4 后续处理建议

- 对于缺失值，根据具体情况选择删除、填充或插值方法
- 对于异常值，考虑删除、转换或保留并在建模时处理
- 根据数据类型转换建议调整数据类型

## 7. 常见问题与解决方案

### 7.1 数据加载失败

**问题**：API无法加载指定的数据文件
**解决方案**：
- 检查文件路径是否正确
- 确保文件格式被支持
- 检查文件是否被其他程序占用

### 7.2 异常值检测结果不准确

**问题**：检测到的异常值过多或过少
**解决方案**：
- 调整异常值检测阈值
- 尝试不同的异常值检测方法
- 考虑数据的分布特征（如是否是偏态分布）

### 7.3 处理大型数据集时性能问题

**问题**：处理大型数据集时速度较慢
**解决方案**：
- 启用数据采样
- 减少需要分析的列
- 关闭不需要的功能（如可视化）

### 7.4 缺失值模式分析结果不明确

**问题**：缺失值模式分析结果难以理解
**解决方案**：
- 生成缺失值热力图进行可视化分析
- 结合业务知识理解缺失值模式
- 考虑使用更高级的缺失值分析方法

## 8. 性能优化

### 8.1 内存优化

- 对于大型数据集，自动启用数据采样
- 使用Dask等分布式计算框架处理超大型数据集
- 优化数据类型以减少内存占用

### 8.2 计算优化

- 使用向量化操作替代循环
- 并行处理多列数据
- 缓存中间结果以避免重复计算

### 8.3 可视化优化

- 对于大型数据集，自动调整可视化参数
- 使用高效的可视化库（如Plotly）
- 支持交互式可视化结果

## 9. 集成与扩展

### 9.1 与其他API的集成

- 与特征工程API无缝集成，预处理结果可直接传递给特征工程API
- 与数据可视化API集成，提供更丰富的可视化结果
- 与建模API集成，预处理后的数据可直接用于建模

### 9.2 扩展功能

- 支持自定义异常值检测方法
- 支持自定义缺失值处理策略
- 支持用户自定义数据质量评估指标
- 提供插件机制，允许用户扩展功能

## 10. 总结

数据预处理API（`get_basic_stats`）是数据分析流程中的基础工具，提供了全面的数据质量检查和初步处理功能。通过该API，用户可以快速了解数据集的基本特征、质量状况和潜在问题，为后续的特征工程和建模提供可靠的数据基础。该API支持多种参数配置，可根据不同的数据类型和分析需求进行调整，同时提供了详细的结果报告和可视化，帮助用户更好地理解和处理数据。