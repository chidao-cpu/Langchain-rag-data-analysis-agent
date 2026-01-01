# 预案性分析API - scenario_analysis

## 1. API概述

预案性分析API（`scenario_analysis`）是数据分析系统中的高级组件，用于对业务场景进行前瞻性模拟和风险评估。该API能够基于历史数据和用户定义的参数，生成多种可能的业务发展路径，并提供量化的结果评估。通过这个API，用户可以模拟各种市场条件、政策变化、风险事件对业务的潜在影响，从而制定更加科学的决策和风险应对策略。

## 2. 功能特性

### 2.1 多维度场景模拟

该API支持从多个维度对业务场景进行模拟分析：

- **市场环境模拟**：模拟不同市场需求、价格波动、竞争态势等外部因素变化
- **政策影响分析**：评估政策调整、法规变化对业务的潜在影响
- **风险事件评估**：分析极端事件、黑天鹅事件对业务的冲击
- **战略决策模拟**：评估不同战略选择（如产能扩张、产品调整、市场进入等）的效果

### 2.2 多种分析方法

API集成了多种先进的分析方法，满足不同场景的需求：

- **敏感性分析**：识别关键参数变化对结果的影响程度
- **压力测试**：模拟极端条件下的业务表现
- **蒙特卡洛模拟**：基于概率分布的随机模拟，生成大量可能的结果
- **决策树分析**：可视化展示不同决策路径的结果
- **情景规划**：构建乐观、中性、悲观等多种情境进行对比分析

### 2.3 智能参数设置

API提供智能参数设置功能，简化用户操作：

- **自动参数建议**：基于历史数据自动推荐合理的参数范围
- **参数依赖检测**：自动检测参数之间的依赖关系，避免不合理的参数组合
- **参数敏感性分析**：自动识别对结果影响最大的参数

### 2.4 与回归模型集成

API与回归分析API无缝集成：

- 直接使用回归模型的预测结果作为模拟基础
- 支持将模拟结果反馈到回归模型进行验证
- 能够利用LSTM等时序模型的预测结果进行动态场景模拟

### 2.5 结果可视化与报告

API提供丰富的结果展示和报告生成功能：

- **概率分布图表**：展示模拟结果的概率分布
- **敏感性热力图**：直观展示参数敏感性
- **情景对比图表**：对比不同情景下的结果差异
- **风险指标报告**：生成VaR（风险价值）、CVaR（条件风险价值）等风险指标
- **决策建议报告**：基于模拟结果提供数据驱动的决策建议

## 3. API参数

### 3.1 基本参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `model` | object | None | 预训练的回归模型对象 |
| `data` | DataFrame | None | 用于模拟的基础数据 |
| `target_column` | string | '利润' | 目标变量列名称 |
| `analysis_type` | string | 'monte_carlo' | 分析类型，支持'sensitivity'、'stress_test'、'monte_carlo'、'decision_tree' |
| `scenario_count` | int | 10000 | 模拟场景数量（仅适用于蒙特卡洛模拟） |

### 3.2 情景参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `scenarios` | dict | None | 用户定义的情景参数，格式为{scenario_name: {param1: value1, param2: value2, ...}} |
| `default_scenario` | string | 'base' | 默认情景名称 |
| `include_optimistic` | bool | True | 是否包含乐观情景 |
| `include_pessimistic` | bool | True | 是否包含悲观情景 |
| `scenario_margin` | float | 0.1 | 乐观/悲观情景的参数波动幅度 |

### 3.3 敏感性分析参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `sensitivity_params` | list | None | 要分析的敏感参数列表 |
| `param_ranges` | dict | None | 参数变化范围，格式为{param: [min, max]} |
| `param_steps` | int | 10 | 参数变化的步数 |
| `sensitivity_metric` | string | 'mse' | 敏感性评估指标 |

### 3.4 蒙特卡洛模拟参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `distributions` | dict | None | 参数的概率分布，格式为{param: {'type': 'normal', 'mean': 0, 'std': 1}} |
| `correlations` | dict | None | 参数之间的相关系数矩阵 |
| `random_seed` | int | 42 | 随机种子，用于结果复现 |
| `confidence_level` | float | 0.95 | 置信水平（用于风险指标计算） |

### 3.5 结果输出参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `output_metrics` | list | ['mean', 'std', 'var_95', 'cvar_95'] | 要输出的指标列表 |
| `generate_report` | bool | True | 是否生成详细报告 |
| `visualize_results` | bool | True | 是否可视化结果 |
| `report_format` | string | 'pdf' | 报告格式，支持'pdf'、'html'、'docx' |
| `save_results` | bool | False | 是否保存模拟结果 |
| `results_path` | string | './scenario_results' | 结果保存路径 |

## 4. API使用方法

### 4.1 基本使用示例

```python
# 导入模块
from src.empirical_analysis.scenario_analysis import ScenarioAnalyzer

# 初始化分析器
scenario_analyzer = ScenarioAnalyzer(
    model=trained_model,  # 预训练的回归模型
    data=analysis_data,   # 分析数据
    target_column='利润'  # 目标变量
)

# 运行蒙特卡洛模拟
results = scenario_analyzer.run_analysis(
    analysis_type='monte_carlo',
    scenario_count=10000,
    confidence_level=0.95
)

# 查看结果
print("模拟结果统计:")
print(f"平均利润: {results['statistics']['mean']:.2f}")
print(f"利润标准差: {results['statistics']['std']:.2f}")
print(f"95%置信区间VaR: {results['risk_metrics']['var_95']:.2f}")
print(f"95%置信区间CVaR: {results['risk_metrics']['cvar_95']:.2f}")
```

### 4.2 敏感性分析示例

```python
# 定义要分析的敏感参数
params_to_analyze = ['市场需求', '产品价格', '生产成本', '营销费用']

# 定义参数变化范围
param_ranges = {
    '市场需求': [0.8, 1.2],   # 80%到120%的基准值
    '产品价格': [0.9, 1.1],   # 90%到110%的基准值
    '生产成本': [0.95, 1.05], # 95%到105%的基准值
    '营销费用': [0.85, 1.15]  # 85%到115%的基准值
}

# 运行敏感性分析
sensitivity_results = scenario_analyzer.run_analysis(
    analysis_type='sensitivity',
    sensitivity_params=params_to_analyze,
    param_ranges=param_ranges,
    param_steps=15
)

# 可视化敏感性结果
scenario_analyzer.visualize_sensitivity(sensitivity_results)
```

### 4.3 自定义情景分析示例

```python
# 定义自定义情景
custom_scenarios = {
    'base': {  # 基准情景
        '市场需求': 1.0,
        '产品价格': 1.0,
        '生产成本': 1.0
    },
    'optimistic': {  # 乐观情景
        '市场需求': 1.15,
        '产品价格': 1.05,
        '生产成本': 0.95
    },
    'pessimistic': {  # 悲观情景
        '市场需求': 0.85,
        '产品价格': 0.95,
        '生产成本': 1.08
    },
    'new_product': {  # 新产品推出情景
        '市场需求': 1.2,
        '产品价格': 1.1,
        '生产成本': 1.05,
        '研发费用': 1.3
    }
}

# 运行自定义情景分析
scenario_results = scenario_analyzer.run_analysis(
    analysis_type='scenario_planning',
    scenarios=custom_scenarios
)

# 生成情景对比报告
scenario_analyzer.generate_scenario_report(scenario_results, report_format='pdf')
```

### 4.4 压力测试示例

```python
# 定义压力测试参数
stress_params = {
    'market_crash': {  # 市场崩盘
        '市场需求': 0.6,
        '产品价格': 0.8,
        '销售渠道': 0.7
    },
    'supply_chain_crisis': {  # 供应链危机
        '生产成本': 1.4,
        '交货时间': 1.8,
        '产品质量': 0.9
    },
    'competitor_attack': {  # 竞争对手攻击
        '市场份额': 0.8,
        '产品价格': 0.9,
        '营销费用': 1.5
    }
}

# 运行压力测试
stress_results = scenario_analyzer.run_analysis(
    analysis_type='stress_test',
    scenarios=stress_params
)

# 查看压力测试结果
for scenario, result in stress_results['scenarios'].items():
    print(f"\n{scenario}:")
    print(f"  利润变化: {result['profit_change']:.2%}")
    print(f"  风险等级: {result['risk_level']}")
    print(f"  恢复时间: {result['recovery_time']}个月")
```

## 5. 返回值说明

### 5.1 基础结果结构

API返回一个包含完整分析结果的字典，主要结构如下：

```python
{
    'analysis_type': 'monte_carlo',  # 分析类型
    'timestamp': '2023-10-15T14:30:00',  # 分析时间戳
    'statistics': {  # 统计指标
        'mean': 156234.56,  # 平均值
        'std': 23456.78,     # 标准差
        'min': 89012.34,     # 最小值
        'max': 210987.65,    # 最大值
        'q1': 138901.23,     # 第一四分位数
        'q2': 156789.01,     # 中位数
        'q3': 174567.89      # 第三四分位数
    },
    'risk_metrics': {  # 风险指标
        'var_95': 105432.10,  # 95%置信区间VaR
        'cvar_95': 92345.67,  # 95%置信区间CVaR
        'max_drawdown': 0.35,  # 最大回撤
        'risk_ratio': 1.87     # 风险回报比
    },
    'simulation_results': [...],  # 详细模拟结果（大型数组）
    'scenarios': {  # 情景分析结果（如果适用）
        'optimistic': {...},
        'base': {...},
        'pessimistic': {...}
    },
    'sensitivity': {...},  # 敏感性分析结果（如果适用）
    'visualizations': {...},  # 可视化结果路径
    'report_path': './scenario_report.pdf'  # 生成的报告路径
}
```

### 5.2 情景分析结果

```python
'scenarios': {
    'scenario_name': {
        'profit': 187654.32,        # 模拟利润
        'profit_change': 0.15,      # 相对于基准的变化比例
        'risk_level': 'low',        # 风险等级
        'key_drivers': ['市场需求', '产品价格'],  # 关键驱动因素
        'confidence': 0.85          # 结果置信度
    }
}
```

### 5.3 敏感性分析结果

```python
'sensitivity': {
    'parameters': ['市场需求', '产品价格', '生产成本'],
    'sensitivity_scores': [0.45, 0.32, 0.23],  # 敏感性分数（越高越敏感）
    'correlations': [0.89, 0.76, -0.65]         # 与结果的相关系数
}
```

## 6. 与其他API的集成

### 6.1 与回归分析API集成

```python
# 首先使用回归分析API训练模型
from src.empirical_analysis.analyzer import EmpiricalAnalyzer

analyzer = EmpiricalAnalyzer(analysis_data, target_column='利润')
analyzer.prepare_data_for_modeling()
models = analyzer.train_regression_models()

# 选择最佳模型
best_model = models['best_model']

# 然后使用预案性分析API进行模拟
from src.empirical_analysis.scenario_analysis import ScenarioAnalyzer

scenario_analyzer = ScenarioAnalyzer(model=best_model, data=analysis_data, target_column='利润')
results = scenario_analyzer.run_analysis(analysis_type='monte_carlo')
```

### 6.2 与特征工程API集成

```python
# 使用特征工程API生成特征
from src.feature_engineering.feature_selector import FeatureSelector

feature_selector = FeatureSelector(analysis_data)
feature_selector.create_time_features()
feature_selector.create_lag_features()
processed_data = feature_selector.get_processed_data()

# 然后使用预案性分析API进行模拟
scenario_analyzer = ScenarioAnalyzer(data=processed_data, target_column='利润')
results = scenario_analyzer.run_analysis(analysis_type='scenario_planning')
```

## 7. 性能优化与最佳实践

### 7.1 性能优化建议

- **合理设置模拟次数**：蒙特卡洛模拟次数越多，结果越精确，但计算时间越长。一般建议10000-50000次模拟
- **使用并行计算**：对于大规模模拟，可启用并行计算加速
- **优化参数范围**：避免设置过宽的参数范围，导致无效模拟
- **使用预采样数据**：对于大型数据集，可先进行采样再进行模拟

### 7.2 最佳实践

- **先进行敏感性分析**：在进行复杂模拟前，先通过敏感性分析识别关键参数
- **结合业务知识**：不要完全依赖模型结果，应结合业务专家知识进行解读
- **定期更新模型**：随着业务环境变化，定期更新基础模型和模拟参数
- **多种方法结合**：综合使用多种分析方法，获得更全面的洞察
- **关注尾部风险**：不仅关注平均结果，更要关注极端情况的风险

## 8. 错误处理与常见问题

### 8.1 常见错误

| 错误类型 | 错误信息 | 解决方案 |
|----------|----------|----------|
| 参数错误 | "Invalid parameter range: min > max" | 检查参数范围设置，确保最小值小于最大值 |
| 模型错误 | "Model not trained properly" | 确保基础模型已经过充分训练 |
| 数据错误 | "Missing required columns" | 检查输入数据是否包含所有必需的列 |
| 计算错误 | "Simulation failed due to numerical instability" | 调整参数范围，避免导致数值不稳定的参数组合 |
| 内存错误 | "Memory error during simulation" | 减少模拟次数，或使用更小的数据集 |

### 8.2 常见问题解答

**Q: 模拟结果与实际情况不符怎么办？**
A: 首先检查基础模型的质量，然后检查模拟参数是否合理，最后考虑是否有重要因素未包含在模拟中。

**Q: 如何选择合适的分析方法？**
A: 根据分析目的选择：敏感性分析适合识别关键参数，压力测试适合评估极端风险，蒙特卡洛模拟适合获得全面的概率分布，情景规划适合对比不同策略。

**Q: 模拟结果的置信度如何？**
A: 模拟结果的置信度取决于基础模型的质量和模拟参数的合理性。建议结合多种方法进行交叉验证。

**Q: 如何解释敏感性分析结果？**
A: 敏感性分数越高，说明该参数对结果的影响越大。正相关表示参数增加会导致结果增加，负相关表示参数增加会导致结果减少。

## 9. 应用案例

### 9.1 零售企业销售预测

某零售企业使用预案性分析API模拟不同促销策略的效果：

- **分析目标**：评估不同促销力度、时间、产品组合对销售的影响
- **使用方法**：蒙特卡洛模拟结合情景规划
- **关键参数**：促销折扣、广告费用、产品组合、竞争对手反应
- **结果**：确定了最优促销策略，预计销售额提升15-20%，同时将风险控制在可接受范围内

### 9.2 制造企业产能扩张决策

某制造企业使用API评估产能扩张方案：

- **分析目标**：评估不同产能扩张规模的投资回报和风险
- **使用方法**：情景规划结合敏感性分析
- **关键参数**：市场需求增长率、投资成本、运营成本、产品价格
- **结果**：选择了中等规模的产能扩张方案，平衡了收益和风险

### 9.3 金融机构风险评估

某金融机构使用API进行风险评估：

- **分析目标**：评估市场波动对投资组合的影响
- **使用方法**：蒙特卡洛模拟结合压力测试
- **关键参数**：利率变化、汇率波动、股票市场表现
- **结果**：确定了合理的风险准备金水平，通过压力测试确保机构在极端情况下仍能正常运营

### 9.4 科技公司新产品上市

某科技公司使用API评估新产品上市策略：

- **分析目标**：评估不同定价、推广策略对新产品销售的影响
- **使用方法**：决策树分析结合情景规划
- **关键参数**：产品定价、推广费用、竞争反应、市场接受度
- **结果**：选择了最优的定价和推广组合，预计新产品上市后6个月内市场份额达到8-10%

## 10. 总结与展望

预案性分析API为企业提供了强大的决策支持工具，能够帮助企业在不确定的环境中做出更加科学的决策。API通过集成多种先进的分析方法，结合智能参数设置和可视化功能，大大降低了复杂分析的门槛。

未来，API将进一步增强：

- 集成更多先进的机器学习算法，提高模拟精度
- 提供更智能的参数优化功能，自动寻找最优决策方案
- 增强与外部数据的集成，实时更新模拟参数
- 提供更丰富的可视化和交互功能，提升用户体验
- 支持更大规模的并行计算，提高分析效率

通过预案性分析API，企业可以更好地应对不确定性，制定更加灵活的战略，提高决策的科学性和准确性，从而在激烈的市场竞争中获得优势。