# 特征工程API - feature_engineering

## 1. API概述

特征工程API（`feature_engineering`）是数据分析和机器学习流程中的核心步骤，负责将原始数据转换为适合模型训练的特征表示。该API集成了多种特征工程技术，包括特征编码、特征缩放、特征选择、特征创建等，旨在提升模型的性能和可解释性。通过自动化和标准化的特征工程流程，用户可以快速构建高质量的特征集，加速建模过程并提高模型效果。

## 2. 功能特性

### 2.1 特征编码

特征编码是将非数值型特征转换为数值型特征的过程，该API支持多种编码方法：

- **独热编码（One-Hot Encoding）**：将类别型特征转换为二进制向量表示
- **标签编码（Label Encoding）**：将类别型特征转换为整数编码
- **目标编码（Target Encoding）**：使用目标变量的统计信息对类别特征进行编码
- **WOE编码（Weight of Evidence）**：基于目标变量的概率比进行编码，适用于二分类问题
- **频率编码（Frequency Encoding）**：使用类别出现的频率进行编码
- **有序编码（Ordinal Encoding）**：保留类别顺序信息的编码方式

### 2.2 特征缩放

特征缩放用于将不同范围的特征转换到相同的尺度，该API支持多种缩放方法：

- **标准缩放（StandardScaler）**：将特征转换为均值为0、标准差为1的分布
- **最小-最大缩放（MinMaxScaler）**：将特征缩放到[0, 1]区间
- **最大绝对值缩放（MaxAbsScaler）**：将特征缩放到[-1, 1]区间
- **鲁棒缩放（RobustScaler）**：使用中位数和四分位距进行缩放，对异常值不敏感
- **幂变换（PowerTransformer）**：将数据转换为更接近正态分布的形式
- **量化缩放（QuantileTransformer）**：将特征转换为均匀分布或正态分布

### 2.3 特征选择

特征选择用于识别和保留对模型预测最有价值的特征，该API支持多种选择方法：

- **统计方法**：相关性分析、卡方检验、ANOVA分析
- **基于模型的方法**：递归特征消除（RFE）、特征重要性排序
- **L1正则化**：使用Lasso回归进行特征选择
- **特征稳定性分析**：评估特征在不同数据集上的稳定性
- **降维方法**：主成分分析（PCA）、线性判别分析（LDA）、t-SNE

### 2.4 特征创建

特征创建用于生成新的衍生特征，该API支持多种创建方法：

- **数学运算**：特征之间的加减乘除等基本运算
- **多项式特征**：生成多项式特征和交互特征
- **分组聚合**：基于类别特征进行分组并计算聚合统计量
- **时间特征**：从时间戳中提取年、月、日、星期等时间特征
- **文本特征**：从文本数据中提取词频、TF-IDF等特征
- **统计特征**：计算滚动统计量（如移动平均、移动标准差）

### 2.5 特征转换

特征转换用于改善特征的分布和性质，该API支持多种转换方法：

- **对数转换**：减少数据的偏度，使分布更接近正态分布
- **平方根转换**：适用于右偏分布的数据
- **Box-Cox转换**：寻找最优的幂变换，使数据更接近正态分布
- **Yeo-Johnson转换**：Box-Cox转换的扩展，支持非正数值
- **分箱处理**：将连续特征转换为离散特征
- **离群值处理**：对异常值进行截断或转换

### 2.6 特征质量评估

特征质量评估用于评估特征的有效性和质量，该API支持多种评估方法：

- **特征重要性**：基于模型的特征重要性排序
- **特征相关性**：评估特征之间的相关性
- **特征区分度**：评估特征对目标变量的区分能力
- **缺失值影响**：评估缺失值对特征质量的影响
- **异常值影响**：评估异常值对特征质量的影响

## 3. API参数

### 3.1 基本参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `data` | DataFrame | None | 输入的Pandas DataFrame对象 |
| `target_column` | string | None | 目标列名称 |
| `numeric_columns` | list | None | 数值型特征列列表 |
| `categorical_columns` | list | None | 类别型特征列列表 |
| `datetime_columns` | list | None | 日期时间型特征列列表 |

### 3.2 编码参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `encoding_method` | string | 'onehot' | 类别特征编码方法，支持'onehot'、'label'、'target'、'woe'、'frequency'、'ordinal' |
| `max_categories` | int | 10 | 独热编码的最大类别数，超过该数量的特征将使用其他编码方法 |
| `target_encoding_smoothing` | float | 1.0 | 目标编码的平滑参数，用于防止过拟合 |

### 3.3 缩放参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `scaling_method` | string | 'standard' | 数值特征缩放方法，支持'standard'、'minmax'、'maxabs'、'robust'、'power'、'quantile' |
| `scaling_range` | tuple | (0, 1) | 最小-最大缩放的目标范围 |
| `power_transform_method` | string | 'yeo-johnson' | 幂变换方法，支持'box-cox'和'yeo-johnson' |

### 3.4 选择参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `feature_selection` | bool | True | 是否执行特征选择 |
| `selection_method` | string | 'auto' | 特征选择方法，支持'auto'、'correlation'、'rfe'、'lasso'、'pca' |
| `n_features` | int | None | 要保留的特征数量 |
| `selection_threshold` | float | 0.01 | 特征选择的阈值 |

### 3.5 创建参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `create_polynomial` | bool | False | 是否生成多项式特征 |
| `polynomial_degree` | int | 2 | 多项式特征的阶数 |
| `create_interaction` | bool | False | 是否生成交互特征 |
| `create_time_features` | bool | True | 是否从日期时间列提取时间特征 |
| `rolling_window` | int | 7 | 滚动统计的窗口大小 |

### 3.6 输出参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `output_format` | string | 'dataframe' | 输出格式，支持'dataframe'、'dict'、'json' |
| `return_transformers` | bool | False | 是否返回拟合后的转换器对象 |
| `return_feature_importance` | bool | True | 是否返回特征重要性信息 |

## 4. 返回值

### 4.1 处理后的数据

```python
{
    'processed_data': pd.DataFrame,  # 特征工程处理后的数据集
    'feature_names': list,           # 处理后的特征名称列表
    'original_feature_names': list,  # 原始特征名称列表
    'feature_mapping': dict          # 特征映射关系
}
```

### 4.2 特征工程配置

```python
{
    'feature_engineering_config': {
        'encoding': {
            'method': 'onehot',
            'max_categories': 10
        },
        'scaling': {
            'method': 'standard'
        },
        'selection': {
            'method': 'auto',
            'n_features': 20
        }
    }
}
```

### 4.3 特征重要性

```python
{
    'feature_importance': {
        'feature1': 0.35,
        'feature2': 0.25,
        'feature3': 0.20,
        # 更多特征...
    },
    'selected_features': ['feature1', 'feature2', 'feature3']
}
```

### 4.4 转换器对象

```python
{
    'transformers': {
        'encoder': OneHotEncoder,    # 编码转换器
        'scaler': StandardScaler,    # 缩放转换器
        'selector': RFECV,           # 选择转换器
        'polynomial': PolynomialFeatures  # 多项式特征转换器
    }
}
```

## 5. 使用示例

### 5.1 基本用法

```python
from src.langchain_agent import LangChainDataAnalysisAgent
import pandas as pd

# 创建代理实例
agent = LangChainDataAnalysisAgent()

# 加载数据
data = pd.read_csv('data/sales_data.csv')

# 执行基本特征工程
result = agent.feature_engineering(
    data=data,
    target_column='利润',
    numeric_columns=['价格', '销量', '成本'],
    categorical_columns=['产品类别', '地区', '促销活动']
)

# 获取处理后的数据
processed_data = result['processed_data']
print("处理后的数据形状:", processed_data.shape)
print("处理后的特征:", result['feature_names'])
```

### 5.2 高级用法

```python
# 执行高级特征工程
result = agent.feature_engineering(
    data=data,
    target_column='利润',
    numeric_columns=['价格', '销量', '成本'],
    categorical_columns=['产品类别', '地区', '促销活动'],
    datetime_columns=['销售日期'],
    
    # 编码参数
    encoding_method='target',
    target_encoding_smoothing=0.5,
    
    # 缩放参数
    scaling_method='robust',
    
    # 选择参数
    feature_selection=True,
    selection_method='rfe',
    n_features=10,
    
    # 创建参数
    create_polynomial=True,
    polynomial_degree=2,
    create_time_features=True,
    rolling_window=14,
    
    # 输出参数
    return_transformers=True,
    return_feature_importance=True
)

# 获取特征重要性
print("特征重要性:", result['feature_importance'])
print("选中的特征:", result['selected_features'])

# 保存处理后的数据
processed_data.to_csv('data/processed_features.csv', index=False)
```

### 5.3 使用保存的转换器

```python
# 保存转换器
import pickle

with open('feature_transformers.pkl', 'wb') as f:
    pickle.dump(result['transformers'], f)

# 加载转换器
with open('feature_transformers.pkl', 'rb') as f:
    transformers = pickle.load(f)

# 使用转换器处理新数据
new_data = pd.read_csv('data/new_sales_data.csv')
processed_new_data = transformers['encoder'].transform(new_data[categorical_columns])
processed_new_data = transformers['scaler'].transform(processed_new_data)
```

## 6. 最佳实践

### 6.1 特征类型识别

- 确保正确识别特征类型（数值型、类别型、日期时间型）
- 对于模糊的特征类型，使用API的自动识别功能
- 手动指定重要的目标列和时间列

### 6.2 编码方法选择

- 对于类别数量较少的特征，优先使用独热编码
- 对于类别数量较多的特征，使用目标编码或频率编码
- 对于有序类别特征，使用有序编码或标签编码
- 对于二分类问题，考虑使用WOE编码

### 6.3 缩放方法选择

- 对于正态分布的数据，使用标准缩放
- 对于有界数据，使用最小-最大缩放
- 对于包含异常值的数据，使用鲁棒缩放
- 对于偏态分布的数据，考虑使用幂变换

### 6.4 特征选择策略

- 优先使用基于模型的特征选择方法
- 结合多种特征选择方法的结果
- 考虑特征的业务意义，不仅仅依赖统计指标
- 保留足够多的特征以避免信息丢失

### 6.5 特征创建建议

- 从时间戳中提取有意义的时间特征
- 生成与业务相关的衍生特征
- 谨慎使用高维多项式特征，避免过拟合
- 考虑使用领域知识创建特征

### 6.6 过拟合预防

- 使用交叉验证评估特征的稳定性
- 对目标编码等可能过拟合的方法使用平滑参数
- 限制多项式特征的阶数
- 避免创建过多的交互特征

## 7. 常见问题与解决方案

### 7.1 内存不足

**问题**：处理大型数据集时出现内存不足错误
**解决方案**：
- 减少生成的特征数量
- 降低多项式特征的阶数
- 使用稀疏矩阵表示高维特征
- 分批处理数据

### 7.2 过拟合

**问题**：特征工程后模型在训练集上表现很好，但在测试集上表现不佳
**解决方案**：
- 减少特征数量
- 增加特征选择的阈值
- 降低多项式特征的阶数
- 对目标编码使用更大的平滑参数

### 7.3 特征重要性不稳定

**问题**：不同运行结果的特征重要性排序不一致
**解决方案**：
- 使用更稳定的特征选择方法
- 增加交叉验证的折数
- 考虑使用集成方法评估特征重要性

### 7.4 时间特征提取错误

**问题**：无法从日期时间列中提取正确的时间特征
**解决方案**：
- 确保日期时间列的格式正确
- 手动指定日期时间列的格式
- 检查是否存在无效的日期时间值

### 7.5 编码后特征数量过多

**问题**：独热编码后特征数量急剧增加
**解决方案**：
- 减少max_categories参数
- 对高基数特征使用其他编码方法
- 增加特征选择的严格程度
- 使用降维方法（如PCA）

## 8. 性能优化

### 8.1 算法优化

- 使用高效的特征工程算法实现
- 对大型数据集使用并行处理
- 缓存中间结果以避免重复计算
- 优先使用基于数组的操作而非循环

### 8.2 内存优化

- 使用适当的数据类型（如使用int8代替int64）
- 对稀疏数据使用稀疏矩阵表示
- 及时释放不再需要的内存
- 使用Dask等分布式计算框架处理超大型数据集

### 8.3 计算优化

- 对大型数据集使用抽样处理
- 减少不必要的特征工程步骤
- 调整参数以减少计算复杂度
- 使用更高效的特征选择方法

## 9. 集成与扩展

### 9.1 与其他API的集成

- 与数据预处理API无缝集成，接受预处理后的数据
- 与可视化API集成，提供特征分布和相关性的可视化
- 与建模API集成，处理后的数据可直接用于模型训练

### 9.2 扩展功能

- 支持自定义特征工程函数
- 提供插件机制，允许用户添加新的特征工程方法
- 支持与外部特征工程库（如Featuretools）集成
- 提供特征工程流水线的可视化编辑器

## 10. 总结

特征工程API（`feature_engineering`）是数据分析和机器学习流程中的关键组件，提供了全面的特征处理功能。通过该API，用户可以自动完成特征编码、缩放、选择和创建等复杂任务，生成高质量的特征集。API支持多种参数配置，可根据不同的数据类型和分析需求进行调整，同时提供了详细的特征重要性评估和质量报告。使用该API可以显著提高特征工程的效率和质量，为后续的建模工作奠定坚实基础。