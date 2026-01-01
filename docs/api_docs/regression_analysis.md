# 回归分析API - regression_analysis

## 1. API概述

回归分析API（`regression_analysis`）是实证分析模块中的核心组件，用于建立和评估各种回归模型，包括传统机器学习模型和深度学习模型（如LSTM）。该API能够自动检测数据类型，针对时序数据自动添加LSTM模型支持，提供全面的模型训练、评估和解释功能。通过这个API，用户可以快速构建和比较多种回归模型，选择最适合业务需求的模型进行预测和分析。

## 2. 功能特性

### 2.1 多模型支持

该API集成了多种回归模型，满足不同的数据分析需求：

- **传统机器学习模型**：
  - 线性回归（Linear Regression）
  - 岭回归（Ridge Regression）
  - Lasso回归（Lasso Regression）
  - 随机森林回归（Random Forest Regression）
  - 梯度提升树回归（Gradient Boosting Regression）

- **深度学习模型**：
  - LSTM（Long Short-Term Memory）模型（针对时序数据自动启用）

### 2.2 时序数据自动检测

该API能够智能检测输入数据是否包含时序特征：

- 自动识别日期/时间列（通过列名和数据类型）
- 检测时间相关特征（如月份、季度、星期几等）
- 针对时序数据自动调整建模策略

### 2.3 LSTM模型集成

当检测到时序数据时，API会自动添加LSTM模型支持：

- 自动将数据转换为LSTM所需的序列格式（样本数, 时间步长, 特征数）
- 智能设置时间步长参数（根据训练集大小自动计算）
- 实现LSTM模型的构建、编译和训练
- 支持早停（Early Stopping）防止过拟合

### 2.4 模型训练与评估

API提供全面的模型训练和评估功能：

- 自动划分训练集和测试集
- 支持交叉验证评估模型性能
- 计算多种评估指标（MSE、MAE、R²等）
- 生成模型性能比较报告

### 2.5 模型选择与保存

API能够自动选择最佳模型并提供保存功能：

- 根据测试集性能自动选择最佳模型
- 支持不同类型模型的保存（传统模型使用pickle，LSTM模型使用h5格式）
- 保存模型训练结果和评估指标

### 2.6 模型解释

API提供模型解释功能，帮助用户理解模型：

- 线性模型的系数分析
- 树模型的特征重要性分析
- 模型性能可视化（如预测值与真实值对比）

## 3. API参数

### 3.1 基本参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `data` | DataFrame | None | 输入的Pandas DataFrame对象 |
| `target_column` | string | '利润' | 目标变量列名称 |
| `test_size` | float | 0.2 | 测试集比例 |
| `random_state` | int | 42 | 随机种子，用于结果复现 |

### 3.2 模型参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `models_to_train` | list | ['linear', 'ridge', 'lasso', 'rf', 'gb'] | 要训练的模型列表，支持'linear'、'ridge'、'lasso'、'rf'、'gb'、'lstm' |
| `include_lstm` | bool | None | 是否包含LSTM模型（默认自动检测时序数据） |
| `time_steps` | int | None | LSTM模型的时间步长（默认自动计算） |

### 3.3 LSTM参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `lstm_units` | list | [64, 32] | LSTM层的神经元数量 |
| `lstm_dropout` | float | 0.2 | LSTM层的Dropout比例 |
| `lstm_epochs` | int | 50 | LSTM模型的训练轮数 |
| `lstm_batch_size` | int | 32 | LSTM模型的批处理大小 |
| `lstm_learning_rate` | float | 0.001 | LSTM模型的学习率 |

### 3.4 评估参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `cv_folds` | int | 5 | 交叉验证的折数 |
| `metrics` | list | ['mse', 'mae', 'r2'] | 评估指标列表，支持'mse'、'mae'、'r2' |

### 3.5 输出参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `output_results` | bool | True | 是否输出模型训练结果 |
| `save_best_model` | bool | True | 是否保存最佳模型 |
| `model_save_path` | string | None | 模型保存路径（默认使用配置文件中的路径） |

## 4. 返回值

### 4.1 模型训练结果

```python
{
    'model_results': pd.DataFrame,  # 所有模型的评估结果
    'best_model_name': str,         # 最佳模型名称
    'best_model': object,           # 最佳模型对象
    'best_model_score': float,      # 最佳模型的评估分数
    'train_predictions': dict,      # 训练集预测结果
    'test_predictions': dict        # 测试集预测结果
}
```

### 4.2 LSTM特定结果

```python
{
    'lstm_history': object,         # LSTM模型的训练历史
    'time_series_detected': bool,   # 是否检测到时序数据
    'time_steps': int,              # LSTM使用的时间步长
    'sequence_shape': tuple         # 序列数据的形状
}
```

### 4.3 模型解释结果

```python
{
    'feature_importance': pd.DataFrame,  # 特征重要性（树模型）
    'coefficients': pd.DataFrame,        # 模型系数（线性模型）
    'performance_comparison': pd.DataFrame  # 模型性能比较
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

# 执行回归分析
result = agent.regression_analysis(
    data=data,
    target_column='利润'
)

# 获取结果
print("模型性能比较:", result['model_results'])
print("最佳模型:", result['best_model_name'])
print("最佳模型分数:", result['best_model_score'])
```

### 5.2 高级用法

```python
# 执行高级回归分析，明确指定模型和参数
result = agent.regression_analysis(
    data=data,
    target_column='利润',
    test_size=0.25,
    random_state=123,
    models_to_train=['linear', 'rf', 'gb', 'lstm'],  # 明确包含LSTM
    time_steps=30,  # 指定时间步长
    lstm_units=[128, 64],  # 调整LSTM神经元数量
    lstm_epochs=100,  # 增加训练轮数
    cv_folds=10,  # 增加交叉验证折数
    save_best_model=True,
    model_save_path='models/best_regression_model.pkl'
)

# 查看LSTM训练历史
if 'lstm_history' in result:
    print("LSTM训练历史:", result['lstm_history'].history['loss'])

# 保存结果
result['model_results'].to_csv('results/regression_results.csv', index=False)
```

### 5.3 时序数据特定用法

```python
# 加载包含时序数据的数据集
time_series_data = pd.read_csv('data/monthly_sales.csv')
time_series_data['销售日期'] = pd.to_datetime(time_series_data['销售日期'])

# 执行回归分析（API会自动检测到时序数据并启用LSTM）
result = agent.regression_analysis(
    data=time_series_data,
    target_column='销售额',
    include_lstm=True  # 明确启用LSTM
)

# 查看时序数据相关信息
print("是否检测到时序数据:", result['time_series_detected'])
print("LSTM时间步长:", result['time_steps'])
print("序列数据形状:", result['sequence_shape'])
```

## 6. 最佳实践

### 6.1 数据准备

- 确保目标变量是数值型
- 对于时序数据，确保日期列格式正确并已转换为datetime类型
- 处理缺失值和异常值（建议使用数据预处理API）
- 进行特征工程以生成有意义的特征（建议使用特征工程API）

### 6.2 模型选择

- 对于简单的线性关系，优先使用线性模型
- 对于复杂的非线性关系，使用树模型（随机森林或梯度提升树）
- 对于时序数据，自动或明确启用LSTM模型
- 建议同时训练多种模型进行比较

### 6.3 LSTM参数调整

- **时间步长**：根据数据的时间频率调整（日度数据可使用30-60天，月度数据可使用12-24个月）
- **神经元数量**：根据数据复杂度调整，一般在32-256之间
- **训练轮数**：使用早停策略防止过拟合，一般设置为50-200轮
- **批处理大小**：根据内存情况调整，一般为32或64
- **学习率**：对于复杂模型可适当降低学习率

### 6.4 模型评估

- 综合考虑多种评估指标（MSE、MAE、R²等）
- 关注模型在测试集上的性能，避免过拟合
- 对于时序数据，特别关注模型的趋势预测能力
- 使用交叉验证评估模型的稳定性

### 6.5 结果解释

- 利用特征重要性或系数分析理解模型决策
- 结合业务知识解释模型结果
- 可视化预测结果与真实值对比
- 考虑模型的可解释性与性能之间的平衡

## 7. 常见问题与解决方案

### 7.1 LSTM模型未自动启用

**问题**：数据集包含时序数据，但API未自动启用LSTM模型

**解决方案**：
- 检查日期列是否已正确转换为datetime类型
- 确保日期列名称包含'date'、'time'或'datetime'等关键字
- 明确设置`include_lstm=True`参数

### 7.2 LSTM训练时间过长

**问题**：LSTM模型训练时间过长

**解决方案**：
- 减少时间步长参数
- 减少LSTM层的神经元数量
- 减少训练轮数
- 增加批处理大小

### 7.3 模型过拟合

**问题**：模型在训练集上表现很好，但在测试集上表现不佳

**解决方案**：
- 增加训练数据量
- 减少模型复杂度（如减少树模型的深度或LSTM的神经元数量）
- 增加正则化（如增加Ridge/Lasso的alpha参数或LSTM的Dropout比例）
- 使用早停策略

### 7.4 特征重要性不合理

**问题**：模型返回的特征重要性与业务直觉不符

**解决方案**：
- 检查特征是否进行了适当的预处理和缩放
- 考虑添加更多与业务相关的特征
- 尝试使用不同的模型（如从树模型切换到线性模型）
- 检查是否存在共线性问题

### 7.5 LSTM预测结果不稳定

**问题**：LSTM模型的预测结果在不同运行之间差异较大

**解决方案**：
- 设置固定的随机种子
- 增加训练数据量
- 增加模型复杂度
- 延长训练时间

## 8. 性能优化

### 8.1 数据处理优化

- 对大型数据集进行采样处理
- 使用更高效的数据格式（如Parquet）
- 对特征进行选择，减少输入维度

### 8.2 模型训练优化

- 对传统模型使用并行计算
- 对LSTM模型使用GPU加速
- 调整批处理大小以充分利用硬件资源
- 使用早停策略避免不必要的训练轮数

### 8.3 内存优化

- 及时释放不再需要的中间变量
- 使用稀疏矩阵表示高维特征
- 对大型模型使用模型压缩技术

## 9. LSTM模型原理与实现

### 9.1 LSTM模型原理

LSTM（Long Short-Term Memory）是一种特殊的循环神经网络（RNN），能够有效处理长期依赖关系：

- 包含输入门、遗忘门和输出门三种门控单元
- 能够选择性地记住或忘记历史信息
- 适合处理时间序列预测、自然语言处理等序列数据任务

### 9.2 数据准备

为了使用LSTM模型，API会自动将数据转换为3D序列格式：

```
(样本数, 时间步长, 特征数)
```

例如，对于包含1000个样本、20个时间步长、5个特征的数据集，序列形状为：

```
(1000, 20, 5)
```

### 9.3 模型结构

API实现的LSTM模型结构如下：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, input_shape=(time_steps, n_features), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])
```

### 9.4 模型训练

LSTM模型使用以下配置进行训练：

```python
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping]
)
```

## 10. 总结

回归分析API（`regression_analysis`）是一个功能强大的工具，能够自动检测数据类型并选择合适的模型，特别是针对时序数据自动添加LSTM模型支持。该API集成了多种回归模型，提供全面的训练、评估和解释功能，帮助用户快速构建高质量的预测模型。通过合理配置参数和遵循最佳实践，用户可以充分利用该API的功能，获得准确的预测结果和有价值的业务洞察。