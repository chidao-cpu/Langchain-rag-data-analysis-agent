# 智能数据分析Agent系统

## 项目简介

智能数据分析Agent系统是一个基于LangChain框架构建的全自动化数据分析平台，集成了RAG（检索增强生成）技术，实现了端到端的数据分析全流程。该系统能够自动完成数据预处理、特征工程、数据可视化、实证分析、预案性分析，并最终生成包含真实数据的专业分析报告。

### 核心价值

- **自动化分析**：无需人工干预，自动完成从数据清洗到报告生成的全流程
- **智能决策**：基于RAG技术从向量库中检索最佳分析方法，辅助Agent做出决策
- **可视化输出**：自动生成多种图表和可视化结果，直观展示数据洞察
- **可扩展架构**：模块化设计，支持轻松扩展新的分析功能和模型
- **专业报告**：生成格式规范、内容全面的Markdown分析报告

## 项目结构

```
AI开发/
├── data/                  # 数据文件目录
│   ├── features_data.csv  # 特征工程后的数据
│   ├── processed_data.csv # 预处理后的数据
│   └── raw_data.csv       # 原始数据
├── docs/                  # 文档目录
│   └── api_docs/          # API文档
├── models/                # 模型文件目录
│   └── best_model.pkl     # 训练好的最佳模型
├── results/               # 分析结果目录
│   ├── visualizations/    # 可视化图表
│   ├── analysis_report.md # 综合分析报告
│   ├── correlation_matrix.csv  # 相关性矩阵
│   ├── descriptive_statistics.csv  # 描述性统计
│   └── model_training_results.csv  # 模型训练结果
├── src/                   # 源代码目录
│   ├── empirical_analysis/    # 实证分析模块
│   ├── feature_engineering/   # 特征工程模块
│   ├── preprocessing/         # 数据预处理模块
│   ├── scenario_analysis/     # 预案性分析模块
│   ├── visualization/         # 数据可视化模块
│   ├── __init__.py            # 包初始化文件
│   ├── config.py              # 配置文件
│   └── langchain_agent.py     # 主Agent实现
├── .env                   # 环境变量配置
├── requirements.txt       # 依赖包列表
└── README.md              # 项目说明文档
```

## 技术栈

| 类别 | 技术 | 版本要求 |
|------|------|----------|
| 核心框架 | LangChain | >=0.1.0 |
| 大语言模型 | DeepSeek API | - |
| 数据处理 | NumPy, Pandas | >=1.26.0, >=2.2.0 |
| 机器学习 | Scikit-learn | >=1.4.0 |
| 数据可视化 | Matplotlib, Seaborn | >=3.8.0, >=0.13.0 |
| 交互式可视化 | Sweetviz | >=2.2.0 |
| 高性能数据 | PyArrow | >=15.0.0 |
| RAG组件 | FAISS, Sentence-transformers | >=1.7.4, >=2.5.0 |
| 开发工具 | Python-dotenv, Jupyter | >=1.0.0 |

## 功能特性

### 1. 数据预处理模块

- ✅ **缺失值处理**：自动检测并处理缺失值
- ✅ **数据类型转换**：智能转换数据类型（日期、分类变量等）
- ✅ **异常值检测**：基于IQR方法检测和处理异常值
- ✅ **衍生特征生成**：自动生成有用的衍生特征

### 2. 特征工程模块

- ✅ **时间特征提取**：从日期时间列提取有用信息
- ✅ **分类特征编码**：自动编码分类变量
- ✅ **数值特征缩放**：标准化或归一化数值特征
- ✅ **特征选择**：基于统计方法选择重要特征

### 3. 数据可视化模块

- ✅ **数值分布分析**：直方图、箱线图等
- ✅ **相关性分析**：热图、散点图矩阵
- ✅ **时间序列分析**：趋势图、季节性分析
- ✅ **分类特征比较**：条形图、饼图等

### 4. 实证分析模块

- ✅ **描述性统计**：均值、中位数、标准差等
- ✅ **相关性分析**：计算和可视化变量间相关性
- ✅ **回归模型**：训练和评估多种回归模型
- ✅ **模型性能比较**：交叉验证和指标对比

### 5. 预案性分析模块

- ✅ **销售增长场景**：模拟不同销售增长情况下的结果
- ✅ **价格变化场景**：分析价格调整对利润的影响
- ✅ **成本降低场景**：评估成本降低策略的效果
- ✅ **组合场景分析**：多因素组合分析

### 6. 智能报告生成

- ✅ **自定义报告结构**：可配置的报告模板
- ✅ **真实数据分析**：所有报告包含真实数据
- ✅ **可视化集成**：自动嵌入生成的图表
- ✅ **业务洞察**：智能生成有价值的业务建议

## 快速开始

### 1. 环境准备

- **Python版本**：3.10或更高版本
- **pip版本**：22.0或更高版本

### 2. 安装依赖

```bash
# 克隆项目或进入项目目录
cd "f:\vscode project 3\AI开发"

# 安装项目依赖
pip install -r requirements.txt
```

### 3. 配置环境变量

创建或修改`.env`文件，配置必要的环境变量：

```
# DeepSeek API密钥配置
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TEMPERATURE=0.1

# 数据文件路径
RAW_DATA_FILE=data/raw_data.csv
PROCESSED_DATA_FILE=data/processed_data.csv
FEATURES_DATA_FILE=data/features_data.csv

# 结果文件路径
ANALYSIS_REPORT=results/analysis_report.md
```

### 4. 运行系统

```bash
# 运行完整数据分析流程
python src/langchain_agent.py
```

### 5. 查看结果

分析完成后，结果将保存在以下位置：

- **数据文件**：`data/`目录下的CSV文件
- **可视化图表**：`results/visualizations/`目录下的图片文件
- **分析报告**：`results/analysis_report.md`文件

## 系统配置

可以通过修改`src/config.py`文件来自定义系统配置：

```python
# 数据处理配置
DATA_PROCESSING_CONFIG = {
    'outlier_detection_method': 'iqr',  # 异常值检测方法
    'missing_value_strategy': 'auto',    # 缺失值处理策略
    'feature_selection_method': 'selectkbest'  # 特征选择方法
}

# 可视化配置
PLOT_CONFIG = {
    'figsize': (12, 8),      # 图表尺寸
    'dpi': 300,              # 图表分辨率
    'style': 'seaborn-v0_8-whitegrid'  # 图表样式
}

# 模型配置
MODEL_CONFIG = {
    'model_types': ['linear', 'random_forest', 'xgboost'],  # 使用的模型类型
    'cv_folds': 5,           # 交叉验证折数
    'random_state': 42       # 随机种子
}
```

## 扩展系统

### 添加新的分析功能

可以通过在`src/`目录下添加新的模块来扩展系统功能：

1. 创建新的Python文件，实现新的分析功能
2. 在`src/langchain_agent.py`中注册新功能作为工具
3. 更新RAG向量库，添加新功能的API文档

### 扩展场景分析

可以在`src/scenario_analysis/scenario_analyzer.py`中添加新的场景分析函数：

```python
def new_scenario_analysis(self, data, parameters):
    # 实现新的场景分析逻辑
    pass
```

## 常见问题

### 1. 报告内容被截断怎么办？

系统已自动设置`max_tokens=8192`（DeepSeek API允许的最大值），确保生成完整的报告内容。

### 2. 运行时出现API错误怎么办？

- 检查`.env`文件中的API密钥和配置是否正确
- 确保网络连接正常
- 查看错误信息，根据提示进行修复

### 3. 如何修改报告的格式和内容？

可以在`src/langchain_agent.py`文件中的`generate_report_with_custom_prompt`方法中修改报告模板。

### 4. 如何添加自定义的可视化图表？

可以在`src/visualization/visualizer.py`中添加新的可视化函数，并在`src/langchain_agent.py`中注册为工具。

## 技术文档

详细的技术文档请参考`docs/`目录下的文件：

- **数据预处理API**：`docs/api_docs/data_preprocessing.md`
- **特征工程API**：`docs/api_docs/feature_engineering.md`
- **数据可视化API**：`docs/api_docs/data_visualization.md`
- **回归分析API**：`docs/api_docs/regression_analysis.md`
- **场景分析报告**：`docs/api_docs/scenario_analysis_report.md`

## 贡献指南

欢迎对项目进行贡献！贡献方式包括：

- **修复bug**：提交Issue和Pull Request
- **增加新功能**：开发新的分析模块或功能
- **优化代码**：提高性能或可读性
- **完善文档**：更新和补充技术文档

## 许可证

本项目采用MIT许可证，详情请参考LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 邮箱：2945128239@qq.com

---

**感谢使用智能数据分析Agent系统！**
