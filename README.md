# 智能数据分析Agent系统

## 项目简介

这是一个基于LangChain框架构建的智能数据分析Agent系统，集成了RAG（检索增强生成）技术，实现了端到端的全流程数据分析。该系统能够自动完成数据预处理、特征工程、数据可视化、实证分析、预案性分析，并最终生成包含真实数据的专业分析报告。

## 关键词
- **算法Agent**：基于LangChain构建的智能数据分析Agent，具备自主决策和工具使用能力
- **RAG技术**：检索增强生成，用于从向量库中检索相关数据分析API信息，辅助决策
- **数据分析全流程**：端到端的数据处理和分析流程，包括数据预处理、特征工程、可视化、实证分析和场景分析

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
│   └── ...                # 其他分析结果文件
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

- **核心框架**：LangChain
- **LLM接口**：DeepSeek API
- **数据处理**：NumPy, Pandas
- **机器学习**：Scikit-learn 
- **数据可视化**：Matplotlib, Seaborn
- **RAG组件**：FAISS, HuggingFace Embeddings
- **其他工具**：python-dotenv, Jupyter

## 功能特性

### 1. 智能数据分析Agent
- 基于LangChain构建的自主Agent，具备工具使用能力
- 能够根据数据集特征动态生成分析策略
- 支持交互式数据分析和查询

### 2. RAG增强的决策系统
- 基于向量库的数据分析API信息检索
- 辅助Agent选择最合适的分析方法和工具
- 提供相关API的详细说明和使用示例

### 3. 全流程数据分析

#### 数据预处理
- 缺失值处理
- 数据类型转换
- 异常值检测和处理
- 衍生特征添加

#### 特征工程
- 时间特征提取
- 分类特征编码
- 数值特征缩放
- 特征选择

#### 数据可视化
- 数值分布分析
- 相关性热图
- 时间序列分析
- 分类特征比较
- 地理分布分析

#### 实证分析
- 描述性统计分析
- 相关性分析
- 回归模型训练与评估
- 模型性能比较

#### 预案性分析
- 销售增长场景分析
- 价格变化场景分析
- 成本降低场景分析
- 组合场景分析

### 4. 智能报告生成
- 基于自定义Prompt的综合报告生成
- 包含真实分析数据的Markdown格式报告
- 突出关键发现和业务建议

## 安装指南

### 1. 环境要求
- Python 3.10+
- pip 22.0+

### 2. 安装依赖

```bash
# 克隆项目
cd f:\vscode project 3\AI开发

# 安装依赖
pip install -r requirements.txt

# 安装额外依赖（用于Markdown表格生成）
pip install tabulate
```

### 3. 配置环境变量

修改`.env`文件，配置必要的环境变量：

```
# API密钥配置
DEEPSEEK_API_KEY=your_api_key_here

# 数据文件路径
RAW_DATA_FILE=data/raw_data.csv
PROCESSED_DATA_FILE=data/processed_data.csv
FEATURES_DATA_FILE=data/features_data.csv

# 结果文件路径
ANALYSIS_REPORT=results/analysis_report.md
```

## 运行指南

### 1. 运行完整数据分析流程

```bash
cd f:\vscode project 3\AI开发
python src/langchain_agent.py
```

这将启动完整的数据分析流程：
1. 数据预处理
2. 特征工程
3. 数据可视化
4. 实证分析
5. 预案性分析
6. 生成综合报告

### 2. 观察分析结果

分析完成后，结果将保存到以下目录：

- **数据文件**：`data/`
  - `raw_data.csv`：原始数据
  - `processed_data.csv`：预处理后的数据
  - `features_data.csv`：特征工程后的数据

- **可视化图表**：`results/visualizations/`
  - 相关性热图、特征重要性图、时间序列图等

- **分析报告**：`results/analysis_report.md`
  - 包含完整的分析结果和业务建议
  - 所有数据均为真实值，无占位符

- **其他结果**：`results/`目录下的CSV文件
  - 模型训练结果
  - 相关性矩阵
  - 场景分析结果

## 使用说明

### 1. 自定义数据分析

系统支持通过修改`src/config.py`文件来自定义分析参数：

```python
# 特征选择方法
FEATURE_SELECTION_METHOD = 'selectkbest'

# 异常值检测方法
OUTLIER_DETECTION_METHOD = 'iqr'

# 可视化配置
PLOT_CONFIG = {
    'figsize': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8-whitegrid'
}
```

### 2. 扩展分析功能

可以通过在`src/`目录下添加新的模块来扩展分析功能，例如：
- 添加新的机器学习模型
- 扩展场景分析的类型
- 增加新的数据可视化图表

### 3. 修改报告模板

报告模板可以在`src/langchain_agent.py`文件中的`generate_report_with_custom_prompt`方法中修改，自定义报告的结构和内容。

## Agent功能详解

### 1. 工具集

Agent系统包含以下工具：
- 数据预处理：处理原始数据
- 特征工程：生成和选择特征
- 数据可视化：生成各种图表
- 实证分析：进行统计和建模分析
- 预案性分析：模拟不同业务场景
- 完整数据分析流程：运行全流程分析
- RAG检索：检索相关数据分析API信息

### 2. RAG机制

系统使用FAISS向量库存储数据分析API信息，当Agent需要选择分析方法时，会通过RAG机制检索相关信息，提高分析的准确性和专业性。

### 3. 报告生成

报告生成使用自定义Prompt和DeepSeek API，确保生成的报告包含真实的分析数据和有价值的业务洞察。

## 项目贡献

欢迎对项目进行贡献！贡献方式包括：
- 修复bug
- 增加新功能
- 优化现有算法
- 完善文档

## 许可证

本项目采用MIT许可证，详情请参考LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱：2945128239@qq.com

---

**感谢使用智能数据分析Agent系统！**