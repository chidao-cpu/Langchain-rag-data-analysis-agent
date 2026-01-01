import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import LLMResult
from typing import Any, List, Optional, Dict, Union
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config
from src.preprocessing.preprocessor import DataPreprocessor
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.visualization.visualizer import DataVisualizer
from src.empirical_analysis.analyzer import EmpiricalAnalyzer
from src.scenario_analysis.scenario_analyzer import ScenarioAnalyzer
warnings.filterwarnings('ignore')

# 自定义OpenAI客户端包装类，用于直接调用DeepSeek API
class OpenAIClientLLM(BaseLanguageModel):
    client: Any
    model: str
    temperature: float
    base_url: str
    api_key: str
    
    def __init__(self,
                 api_key: str='your_api_key_here',
                 base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-chat",
                 temperature: float = 0.1):
        # 初始化OpenAI客户端，只传递必要参数，不接受任何额外参数
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 调用父类初始化，传递所有必要参数
        super().__init__(
            client=client,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature
        )
    
    def _generate(self,
                 prompts: List[str],
                 stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            # 只使用必要的API参数，不接受任何额外参数
            api_kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "stop": stop
            }
                
            response = self.client.chat.completions.create(**api_kwargs)
            text = response.choices[0].message.content
            generations.append([{"text": text}])
        return LLMResult(generations=generations)
    
    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        # 只使用必要的API参数，不接受任何额外参数
        api_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stop": stop
        }
            
        response = self.client.chat.completions.create(**api_kwargs)
        return response.choices[0].message.content
    
    def generate_prompt(self,
                       prompts: List[Dict[str, Any]],
                       stop: Optional[List[str]] = None,
                       **kwargs: Any) -> LLMResult:
        # 将prompt转换为字符串
        string_prompts = []
        for prompt in prompts:
            if isinstance(prompt, dict):
                # 处理messages格式
                string_prompt = ""
                for msg in prompt.get("messages", []):
                    string_prompt += f"{msg['role']}: {msg['content']}\n"
                string_prompts.append(string_prompt)
            else:
                string_prompts.append(str(prompt))
        # 调用_generate时不传递任何额外参数
        return self._generate(string_prompts, stop)
    
    def invoke(self, input: Union[str, List[BaseMessage], Any], stop: Optional[List[str]] = None, config: Optional[Any] = None, **kwargs: Any) -> Any:
            
        if isinstance(input, str):
            return self._call(input, stop=stop)
        elif isinstance(input, list):
            # 将消息列表转换为字符串
            prompt = ""
            for msg in input:
                # 根据消息类型获取角色
                if hasattr(msg, 'role'):
                    role = msg.role
                elif 'SystemMessage' in type(msg).__name__:
                    role = 'system'
                elif 'HumanMessage' in type(msg).__name__:
                    role = 'user'
                elif 'AIMessage' in type(msg).__name__:
                    role = 'assistant'
                else:
                    role = 'unknown'
                prompt += f"{role}: {msg.content}\n"
            return self._call(prompt, stop=stop)
        elif hasattr(input, 'to_messages'):
            # 处理ChatPromptValue类型的输入
            messages = input.to_messages()
            prompt = ""
            for msg in messages:
                # 根据消息类型获取角色
                if hasattr(msg, 'role'):
                    role = msg.role
                elif 'SystemMessage' in type(msg).__name__:
                    role = 'system'
                elif 'HumanMessage' in type(msg).__name__:
                    role = 'user'
                elif 'AIMessage' in type(msg).__name__:
                    role = 'assistant'
                else:
                    role = 'unknown'
                prompt += f"{role}: {msg.content}\n"
            return self._call(prompt, stop=stop)
        elif hasattr(input, 'to_string'):
            # 处理其他PromptValue类型的输入
            prompt = input.to_string()
            return self._call(prompt, stop=stop)
        else:
            raise ValueError(f"Unexpected input type: {type(input)}")
    
    def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any):
        raise NotImplementedError("Async generation not implemented")
    
    async def agenerate_prompt(self, prompts: List[Dict[str, Any]], stop: Optional[List[str]] = None, **kwargs: Any):
        raise NotImplementedError("Async generate_prompt not implemented")
    
    def _aget(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any):
        raise NotImplementedError("Async call not implemented")
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "base_url": self.client.base_url
        }
    
    @property
    def _llm_type(self) -> str:
        return "openai-client-llm"
    
    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "OpenAIClientLLM":
        """绑定工具到LLM，返回新的LLM实例
        
        Args:
            tools: 要绑定的工具列表
            **kwargs: 额外参数
            
        Returns:
            绑定了工具的新LLM实例
        """

        return self


class LangChainDataAnalysisAgent:
    def __init__(self):
        self.config = Config()
        self.preprocessor = None
        self.feature_engineer = None
        self.visualizer = None
        self.analyzer = None
        self.scenario_analyzer = None
        self.vector_db = None
        self.embeddings = None
        
        # 初始化各个模块
        self._initialize_modules()
        
        # 分析数据集结构
        self.analyze_dataset()
        
        # 初始化向量库
        self._initialize_vector_db()
        
        # 生成动态提示词
        self.prompt = self.generate_dynamic_prompt()
        
        # 创建工具列表
        self.tools = self._create_tools()
    
    def _initialize_modules(self):
        """初始化各个数据分析模块"""
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.visualizer = DataVisualizer()
        self.analyzer = EmpiricalAnalyzer()
        self.scenario_analyzer = ScenarioAnalyzer()
    
    def _initialize_vector_db(self):
        """初始化向量库，用于存储和检索数据分析报告API"""
        print("=== 初始化向量库 ===")
        
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL)
        
        # 使用临时内存向量库，不进行持久化存储
        print("创建内存向量库（不进行持久化）")
        
        # 初始化文档分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,  # 每个文档块的大小 - 减小块大小以增加块数
            chunk_overlap=25,  # 文档块之间的重叠部分
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 从文件加载初始文档
        initial_documents = []
        api_docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'api_docs')
        
        # 定义文档元数据映射
        doc_metadata = {
            'data_preprocessing.md': {"category": "数据预处理", "api_name": "data_preprocessing"},
            'feature_engineering.md': {"category": "特征工程", "api_name": "feature_engineering"},
            'data_visualization.md': {"category": "数据可视化", "api_name": "data_visualization"},
            'regression_analysis.md': {"category": "实证分析", "api_name": "regression_analysis"},
            'scenario_analysis.md': {"category": "预案性分析", "api_name": "scenario_analysis"}
        }
        
        # 读取所有API文档文件
        for filename, metadata in doc_metadata.items():
            file_path = os.path.join(api_docs_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 创建初始文档对象
                    document = Document(page_content=content, metadata=metadata)
                    # 将文档分割成小块
                    doc_chunks = text_splitter.split_documents([document])
                    # 添加到初始文档列表
                    initial_documents.extend(doc_chunks)
                print(f"已加载并分割文档: {filename} (共 {len(doc_chunks)} 个块)")
            else:
                print(f"警告: 文档文件不存在: {file_path}")
        
        # 创建内存向量库（不保存到磁盘）
        self.vector_db = FAISS.from_documents(initial_documents, self.embeddings)
        print(f"向量库创建成功（内存模式），共加载 {len(initial_documents)} 个文档块")
        

    
    def analyze_dataset(self, data_file=None):
        """分析数据集结构，识别数据类型、日期列、可能的目标列等"""
        file_path = data_file or self.config.RAW_DATA_FILE
        if not os.path.exists(file_path):
            print(f"警告: 数据文件不存在 - {file_path}")
            return {}
        
        data = pd.read_csv(file_path)
        dataset_info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'data_types': data.dtypes.astype(str).to_dict(),
            'numeric_cols': data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_cols': data.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_cols': [],
            'possible_target_cols': [],
            'missing_values': data.isnull().sum().to_dict()
        }
        
        # 自动识别日期列
        for col in data.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'datetime' in col.lower():
                try:
                    pd.to_datetime(data[col])
                    dataset_info['datetime_cols'].append(col)
                except:
                    pass
        
        # 识别可能的目标列（数值型列，不是明显的ID或时间列）
        for col in dataset_info['numeric_cols']:
            if col not in dataset_info['datetime_cols'] and not ('id' in col.lower() or 'index' in col.lower()):
                dataset_info['possible_target_cols'].append(col)
        
        self.dataset_info = dataset_info
        print(f"数据集分析完成: {dataset_info}")
        return dataset_info
    
    def generate_dynamic_prompt(self):
        """根据数据集特征生成动态提示词，选择合适的分析模板"""
        if not hasattr(self, 'dataset_info'):
            self.analyze_dataset()
        
        dataset_info = self.dataset_info
        
        # 使用RAG检索相关数据分析API信息
        query = f"针对数据集{dataset_info['shape']}进行数据分析，包含{dataset_info['columns']}列，其中数值型列{dataset_info['numeric_cols']}，类别型列{dataset_info['categorical_cols']}，日期时间列{dataset_info['datetime_cols']}"
        rag_info = self.retrieve_rag_info(query) if hasattr(self, 'retrieve_rag_info') and self.vector_db else ""
        
        # 分析数据集类型
        has_time_series = len(dataset_info['datetime_cols']) > 0
        has_numeric = len(dataset_info['numeric_cols']) > 0
        has_categorical = len(dataset_info['categorical_cols']) > 0
        has_target = len(dataset_info['possible_target_cols']) > 0
        
        # 选择合适的提示词模板
        if has_time_series:
            # 时间序列数据模板
            prompt_template = f"""你是一个专业的数据分析助手，需要根据以下时间序列数据集信息完成数据分析任务：
        
数据集基本信息：
- 数据形状：{dataset_info['shape']}
- 列名：{', '.join(dataset_info['columns'])}
- 数值型列：{', '.join(dataset_info['numeric_cols']) if dataset_info['numeric_cols'] else '无'}
- 类别型列：{', '.join(dataset_info['categorical_cols']) if dataset_info['categorical_cols'] else '无'}
- 日期时间列：{', '.join(dataset_info['datetime_cols']) if dataset_info['datetime_cols'] else '无'}
- 可能的目标列：{', '.join(dataset_info['possible_target_cols']) if dataset_info['possible_target_cols'] else '无'}
        
以下是通过RAG检索到的相关数据分析API信息：
{rag_info}
        
请根据以下工具列表和数据集信息，选择合适的工具完成数据分析任务：

1. 数据预处理：对原始数据进行预处理，包括处理缺失值、异常值、转换数据类型、添加衍生特征等。
2. 特征工程：对预处理后的数据进行特征工程，包括编码分类特征、缩放数值特征、选择重要特征等。
3. 数据可视化：生成各种数据可视化图表，包括分布直方图、相关性热图、时间序列图等。
4. 实证分析：进行实证分析，包括描述性统计、相关性分析、假设检验、回归建模等。
5. 预案性分析：根据数据集特征进行相应的场景分析。
6. 完整数据分析流程：运行完整的数据分析流程。
7. RAG检索：从向量库中检索相关的数据分析报告API信息，帮助选择合适的分析方法。

针对时间序列数据，建议关注以下分析方向：
- 时间序列趋势分析
- 季节性模式识别
- 周期性分析
- 预测建模
- 异常检测

请确保分析结果能够提供有价值的洞察和建议。"""
        
        elif has_categorical and has_numeric:
            # 混合数据类型模板
            prompt_template = f"""你是一个专业的数据分析助手，需要根据以下混合类型数据集信息完成数据分析任务：
        
数据集基本信息：
- 数据形状：{dataset_info['shape']}
- 列名：{', '.join(dataset_info['columns'])}
- 数值型列：{', '.join(dataset_info['numeric_cols']) if dataset_info['numeric_cols'] else '无'}
- 类别型列：{', '.join(dataset_info['categorical_cols']) if dataset_info['categorical_cols'] else '无'}
- 日期时间列：{', '.join(dataset_info['datetime_cols']) if dataset_info['datetime_cols'] else '无'}
- 可能的目标列：{', '.join(dataset_info['possible_target_cols']) if dataset_info['possible_target_cols'] else '无'}
        
以下是通过RAG检索到的相关数据分析API信息：
{rag_info}
        
请根据以下工具列表和数据集信息，选择合适的工具完成数据分析任务：

1. 数据预处理：对原始数据进行预处理，包括处理缺失值、异常值、转换数据类型、添加衍生特征等。
2. 特征工程：对预处理后的数据进行特征工程，包括编码分类特征、缩放数值特征、选择重要特征等。
3. 数据可视化：生成各种数据可视化图表，包括分布直方图、相关性热图、时间序列图等。
4. 实证分析：进行实证分析，包括描述性统计、相关性分析、假设检验、回归建模等。
5. 预案性分析：根据数据集特征进行相应的场景分析。
6. 完整数据分析流程：运行完整的数据分析流程。
7. RAG检索：从向量库中检索相关的数据分析报告API信息，帮助选择合适的分析方法。

针对混合类型数据，建议关注以下分析方向：
- 不同类别间的数值差异分析
- 特征间的相关性分析
- 分类建模
- 聚类分析
- 特征重要性评估

请确保分析结果能够提供有价值的洞察和建议。"""
        
        elif has_numeric and not has_categorical:
            # 纯数值数据模板
            prompt_template = f"""你是一个专业的数据分析助手，需要根据以下数值型数据集信息完成数据分析任务：
        
数据集基本信息：
- 数据形状：{dataset_info['shape']}
- 列名：{', '.join(dataset_info['columns'])}
- 数值型列：{', '.join(dataset_info['numeric_cols']) if dataset_info['numeric_cols'] else '无'}
- 类别型列：{', '.join(dataset_info['categorical_cols']) if dataset_info['categorical_cols'] else '无'}
- 日期时间列：{', '.join(dataset_info['datetime_cols']) if dataset_info['datetime_cols'] else '无'}
- 可能的目标列：{', '.join(dataset_info['possible_target_cols']) if dataset_info['possible_target_cols'] else '无'}
        
以下是通过RAG检索到的相关数据分析API信息：
{rag_info}
        
请根据以下工具列表和数据集信息，选择合适的工具完成数据分析任务：

1. 数据预处理：对原始数据进行预处理，包括处理缺失值、异常值、转换数据类型、添加衍生特征等。
2. 特征工程：对预处理后的数据进行特征工程，包括编码分类特征、缩放数值特征、选择重要特征等。
3. 数据可视化：生成各种数据可视化图表，包括分布直方图、相关性热图、时间序列图等。
4. 实证分析：进行实证分析，包括描述性统计、相关性分析、假设检验、回归建模等。
5. 预案性分析：根据数据集特征进行相应的场景分析。
6. 完整数据分析流程：运行完整的数据分析流程。
7. RAG检索：从向量库中检索相关的数据分析报告API信息，帮助选择合适的分析方法。

针对纯数值数据，建议关注以下分析方向：
- 数值分布分析
- 变量间的相关性分析
- 回归建模
- 降维分析
- 异常检测

请确保分析结果能够提供有价值的洞察和建议。"""
        
        else:
            # 默认模板
            prompt_template = f"""你是一个专业的数据分析助手，需要根据以下数据集信息完成数据分析任务：
        
数据集基本信息：
- 数据形状：{dataset_info['shape']}
- 列名：{', '.join(dataset_info['columns'])}
- 数值型列：{', '.join(dataset_info['numeric_cols']) if dataset_info['numeric_cols'] else '无'}
- 类别型列：{', '.join(dataset_info['categorical_cols']) if dataset_info['categorical_cols'] else '无'}
- 日期时间列：{', '.join(dataset_info['datetime_cols']) if dataset_info['datetime_cols'] else '无'}
- 可能的目标列：{', '.join(dataset_info['possible_target_cols']) if dataset_info['possible_target_cols'] else '无'}
        
以下是通过RAG检索到的相关数据分析API信息：
{rag_info}
        
请根据以下工具列表和数据集信息，选择合适的工具完成数据分析任务：

1. 数据预处理：对原始数据进行预处理，包括处理缺失值、异常值、转换数据类型、添加衍生特征等。
2. 特征工程：对预处理后的数据进行特征工程，包括编码分类特征、缩放数值特征、选择重要特征等。
3. 数据可视化：生成各种数据可视化图表，包括分布直方图、相关性热图、时间序列图等。
4. 实证分析：进行实证分析，包括描述性统计、相关性分析、假设检验、回归建模等。
5. 预案性分析：根据数据集特征进行相应的场景分析。
6. 完整数据分析流程：运行完整的数据分析流程，包括数据预处理、特征工程、数据可视化、实证分析和预案性分析，并生成综合报告。
7. RAG检索：从向量库中检索相关的数据分析报告API信息，帮助选择合适的分析方法。

请根据数据集的具体情况，选择合适的工具和参数进行分析。例如：
- 如果有日期时间列，可以考虑添加时间相关的衍生特征
- 如果有多个数值型列，可以进行相关性分析和回归建模
- 如果有类别型列，可以进行分类特征分析和编码

请确保分析结果能够提供有价值的洞察和建议。"""
        
        return prompt_template
    
    def _create_tools(self):
        """创建langchain工具列表"""
        tools = [
            Tool(
                name="数据预处理",
                func=self.run_preprocessing,
                description="对原始数据进行预处理，包括处理缺失值、异常值、转换数据类型、添加衍生特征等。输入参数为可选的数据文件路径。"
            ),
            Tool(
                name="特征工程",
                func=self.run_feature_engineering,
                description="对预处理后的数据进行特征工程，包括编码分类特征、缩放数值特征、选择重要特征等。输入参数为可选的目标列名。"
            ),
            Tool(
                name="数据可视化",
                func=self.run_visualization,
                description="生成各种数据可视化图表，包括分布直方图、相关性热图、时间序列图、分类特征比较图等。"
            ),
            Tool(
                name="实证分析",
                func=self.run_empirical_analysis,
                description="进行实证分析，包括描述性统计、相关性分析、假设检验、回归建模、模型评估和解释等。"
            ),
            Tool(
                name="预案性分析",
                func=self.run_scenario_analysis,
                description="根据数据集特征进行相应的场景分析，包括趋势预测、影响因素分析等。"
            ),
            Tool(
                name="完整数据分析流程",
                func=self.run_full_analysis,
                description="运行完整的数据分析流程，包括数据预处理、特征工程、数据可视化、实证分析和预案性分析。"
            ),
            Tool(
                name="RAG检索",
                func=self.retrieve_rag_info,
                description="从向量库中检索相关的数据分析报告API信息，帮助选择合适的分析方法。输入参数为查询字符串。"
            )
        ]
        return tools
    

    
    def run_preprocessing(self, data_file=None):
        """运行数据预处理"""
        print("=== 开始数据预处理 ===")
        if data_file:
            self.preprocessor.load_data(data_file)
        else:
            self.preprocessor.load_data()
        
        self.preprocessor.handle_missing_values()
        self.preprocessor.convert_data_types()
        self.preprocessor.detect_outliers()
        self.preprocessor.remove_outliers()
        self.preprocessor.add_derived_columns()
        self.preprocessor.save_processed_data()
        
        print("=== 数据预处理完成 ===")
        return "数据预处理已完成，预处理后的数据已保存到: " + self.config.PROCESSED_DATA_FILE
    
    def run_feature_engineering(self, target_col=None):
        """运行特征工程"""
        print("=== 开始特征工程 ===")
        self.feature_engineer.load_processed_data()
        self.feature_engineer.add_time_based_features()
        self.feature_engineer.encode_categorical_features(method='onehot')
        self.feature_engineer.scale_numerical_features(method='standard')
        self.feature_engineer.select_features(target_col=target_col, method=self.config.FEATURE_SELECTION_METHOD)
        self.feature_engineer.save_features_data()
        
        print("=== 特征工程完成 ===")
        return "特征工程已完成，特征工程后的数据已保存到: " + self.config.FEATURES_DATA_FILE
    
    def run_visualization(self):
        """运行数据可视化"""
        print("=== 开始数据可视化 ===")
        self.visualizer.run_all_visualizations()
        
        print("=== 数据可视化完成 ===")
        return "数据可视化已完成，所有图表已保存到: " + self.config.VISUALIZATIONS_DIR
    
    def run_empirical_analysis(self):
        """运行实证分析"""
        print("=== 开始实证分析 ===")
        results = self.analyzer.run_analysis()
        
        print("=== 实证分析完成 ===")
        return "实证分析已完成，分析结果已保存到: " + self.config.RESULTS_DIR
    
    def run_scenario_analysis(self):
        """运行预案性分析"""
        print("=== 开始预案性分析 ===")
        results = self.scenario_analyzer.run_scenario_analysis()
        
        print("=== 预案性分析完成 ===")
        return "预案性分析已完成，分析结果已保存到: " + self.config.RESULTS_DIR
    
    def retrieve_rag_info(self, query):
        """使用RAG从向量库中检索相关的数据分析报告API信息"""
        print(f"=== 开始RAG检索: {query} ===")
        
        if not self.vector_db:
            print("向量库未初始化，无法执行检索")
            return "向量库未初始化，无法执行检索"
        
        try:
            # 执行相似度搜索
            results = self.vector_db.similarity_search(query, k=200)
            
            # 格式化检索结果
            retrieved_info = "\n".join([
                f"- API: {result.metadata.get('api_name', '未知')} (分类: {result.metadata.get('category', '未知')})\n  描述: {result.page_content}"
                for result in results
            ])
            
            print(f"=== RAG检索完成，找到 {len(results)} 条相关信息 ===")
            return f"已找到以下相关数据分析API信息:\n{retrieved_info}"
            
        except Exception as e:
            print(f"RAG检索过程中出现错误: {e}")
            return f"RAG检索过程中出现错误: {e}"
    
    def run_full_analysis(self, data_file=None):
        """运行完整数据分析流程"""
        print("="*60)
        print("开始全链路数据分析流程")
        print("="*60)
        

        # 1. 数据预处理
        self.run_preprocessing(data_file)
        
        # 2. 特征工程
        self.run_feature_engineering()
        
        # 3. 数据可视化
        self.run_visualization()
        
        # 4. 实证分析
        self.run_empirical_analysis()
        
        # 5. 预案性分析
        self.run_scenario_analysis()
        
        # 6. 使用自定义prompt生成综合报告
        self.generate_report_with_custom_prompt()
        
        print("\n" + "="*60)
        print("全链路数据分析流程完成")
        print("="*60)
        print("\n分析结果已保存到以下目录:")
        print(f"- 数据文件: {self.config.DATA_DIR}")
        print(f"- 模型文件: {self.config.MODELS_DIR}")
        print(f"- 结果文件: {self.config.RESULTS_DIR}")
        print(f"- 可视化图表: {self.config.VISUALIZATIONS_DIR}")
        
        return "完整数据分析流程已完成，所有结果已保存。"
        

    def generate_report_with_custom_prompt(self):
        """使用自定义prompt生成综合分析报告"""
        print("=== 开始使用自定义prompt生成综合报告 ===")
        
        report_file = self.config.ANALYSIS_REPORT
        
        # 收集所有分析结果的信息
        analysis_results = {
            "data_overview": {
                "source": self.config.RAW_DATA_FILE,
                "shape": self.dataset_info.get('shape', '未知') if hasattr(self, 'dataset_info') else '未知',
                "columns": self.dataset_info.get('columns', []) if hasattr(self, 'dataset_info') else []
            },
            "preprocessing": {
                "steps": ["加载数据", "处理缺失值", "转换数据类型", "检测和处理异常值", "添加衍生列", "保存预处理后的数据"],
                "result_file": self.config.PROCESSED_DATA_FILE
            },
            "feature_engineering": {
                "steps": ["编码分类特征", "缩放数值特征", "选择重要特征", "添加时间特征", "保存特征工程后的数据"],
                "result_file": self.config.FEATURES_DATA_FILE
            },
            "visualization": {
                "chart_types": ["数值型特征分布", "相关性热图", "时间序列分析", "分类特征比较", "特征散点图", "促销活动效果", "地区分布"],
                "result_dir": self.config.VISUALIZATIONS_DIR
            },
            "empirical_analysis": {
                "content": ["描述性统计分析", "相关性分析", "假设检验", "回归模型训练与评估", "模型解释"],
                "result_dir": self.config.RESULTS_DIR
            },
            "scenario_analysis": {
                "content": ["销售增长场景", "价格变化场景", "成本降低场景", "组合场景"],
                "result_dir": self.config.RESULTS_DIR
            }
        }
        
        # 收集实证分析和预案性分析的实际结果数据
        empirical_data = ""
        scenario_data = ""
        
        # 收集实证分析数据
        if hasattr(self.analyzer, 'results'):
            empirical_data = f"\n### 实证分析实际结果\n\n"
            empirical_data += "#### 模型性能比较\n"
            empirical_data += self.analyzer.results.to_markdown(index=False) + "\n"
            
        # 收集预案性分析数据
        if hasattr(self.scenario_analyzer, 'baseline'):
            scenario_data = f"\n### 预案性分析实际结果\n\n"
            
            # 基线数据
            scenario_data += "#### 当前基线数据\n"
            scenario_data += f"- 平均价格: {self.scenario_analyzer.baseline['平均价格']:.2f}\n"
            scenario_data += f"- 平均销量: {self.scenario_analyzer.baseline['平均销量']:.2f}\n"
            scenario_data += f"- 平均成本: {self.scenario_analyzer.baseline['平均成本']:.2f}\n"
            scenario_data += f"- 平均利润: {self.scenario_analyzer.baseline['平均利润']:.2f}\n"
            scenario_data += f"- 总销售额: {self.scenario_analyzer.baseline['总销售额']:.2f}\n"
            scenario_data += f"- 总利润: {self.scenario_analyzer.baseline['总利润']:.2f}\n"
            
            # 销售增长场景
            sales_results = self.scenario_analyzer.sales_results if hasattr(self.scenario_analyzer, 'sales_results') else None
            if sales_results is not None:
                scenario_data += "\n#### 销售增长场景\n"
                scenario_data += sales_results.to_markdown(index=False) + "\n"
            
            # 价格变化场景
            price_results = self.scenario_analyzer.price_results if hasattr(self.scenario_analyzer, 'price_results') else None
            if price_results is not None:
                scenario_data += "\n#### 价格变化场景\n"
                scenario_data += price_results.to_markdown(index=False) + "\n"
            
            # 成本降低场景
            cost_results = self.scenario_analyzer.cost_results if hasattr(self.scenario_analyzer, 'cost_results') else None
            if cost_results is not None:
                scenario_data += "\n#### 成本降低场景\n"
                scenario_data += cost_results.to_markdown(index=False) + "\n"
            
            # 组合场景
            combined_results = self.scenario_analyzer.combined_results if hasattr(self.scenario_analyzer, 'combined_results') else None
            if combined_results is not None:
                scenario_data += "\n#### 组合场景\n"
                scenario_data += combined_results.to_markdown(index=False) + "\n"
        
        # 构建自定义prompt
        custom_prompt = f"""请根据以下数据分析结果生成一份专业的综合数据分析报告：

## 数据分析结果概览

### 数据概况
- 数据来源: {analysis_results['data_overview']['source']}
- 数据形状: {analysis_results['data_overview']['shape']}
- 数据列: {', '.join(analysis_results['data_overview']['columns'])}

### 数据预处理
- 预处理步骤: {', '.join(analysis_results['preprocessing']['steps'])}
- 预处理结果文件: {analysis_results['preprocessing']['result_file']}

### 特征工程
- 特征工程步骤: {', '.join(analysis_results['feature_engineering']['steps'])}
- 特征工程结果文件: {analysis_results['feature_engineering']['result_file']}

### 数据可视化
- 可视化图表类型: {', '.join(analysis_results['visualization']['chart_types'])}
- 可视化结果目录: {analysis_results['visualization']['result_dir']}

### 实证分析
- 分析内容: {', '.join(analysis_results['empirical_analysis']['content'])}
- 分析结果目录: {analysis_results['empirical_analysis']['result_dir']}

### 预案性分析
- 分析内容: {', '.join(analysis_results['scenario_analysis']['content'])}
- 分析结果目录: {analysis_results['scenario_analysis']['result_dir']}

{empirical_data}
{scenario_data}

## 报告要求
1. 报告结构清晰，包含标题、摘要、主要部分和结论
2. 每个部分应有详细的分析和解释
3. 突出关键发现和洞察
4. 提供具体的业务建议
5. 语言专业但易于理解
6. 报告中必须包含上述实际的分析结果数据，不能用字母代替真实数值
请生成一份完整的Markdown格式的综合数据分析报告。"""
        
        # 生成报告内容
        if hasattr(self, 'use_real_llm') and self.use_real_llm and hasattr(self, 'api_key') and self.api_key:
            # 使用自定义的OpenAIClientLLM类直接调用DeepSeek API
            llm = OpenAIClientLLM(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat",
                temperature=0.1
            )
            report_content = llm._call(custom_prompt)
        
        # 将报告内容保存到文件
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"=== 报告已保存至: {report_file} ===")
            return f"报告已生成并保存至: {report_file}"
        except Exception as e:
            print(f"=== 报告保存失败: {e} ===")
            return f"报告生成失败: {e}"
 
    def create_agent_executor(self, use_real_llm=False, api_key='your_api_key_here'):
        """创建并返回Agent执行器
        
        Args:
            use_real_llm: 是否使用真实的大模型接口
            api_key: 大模型API密钥
        """

        # 保存配置信息
        self.use_real_llm = use_real_llm
        self.api_key = api_key
        
        # 初始化大模型
        if use_real_llm and api_key:
            # 使用自定义的OpenAIClientLLM类直接调用DeepSeek API
            llm = OpenAIClientLLM(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat",
                temperature=0.1
            )
 

        # 动态生成提示词内容
        dynamic_prompt_content = self.prompt
        
        # 创建提示词模板，包含动态生成的提示词内容
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", dynamic_prompt_content),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # 创建Agent
        agent = create_tool_calling_agent(llm, self.tools, prompt_template)
        
        # 创建并返回Agent执行器
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        return agent_executor

def main():
    """主函数"""
    print("="*60)
    print("基于LangChain的数据分析Agent")
    print("="*60)
    
    try:
        # 创建Agent实例
        agent = LangChainDataAnalysisAgent()
        
        # 默认使用真实DeepSeek模型
        use_real_llm = True
        # 直接提供API密钥
        api_key = "your_api_key_here"

        # 保存配置信息
        agent.use_real_llm = use_real_llm
        agent.api_key = api_key
        
        # 开始完整的数据分析流程
        print("\n" + "="*60)
        print("开始完整的数据分析流程:")
        print("1. 数据预处理")
        print("2. 特征工程")
        print("3. 数据可视化")
        print("4. 实证分析")
        print("5. 预案性分析")
        print("6. 生成综合报告")
        print("="*60)
        print("\n开始运行完整数据分析流程...")
        
        # 直接调用完整分析流程
        result = agent.run_full_analysis()
        
        print("\n" + "="*60)
        print("数据分析流程完成！")
        print(result)
        print("\n感谢使用数据分析Agent！")
        print("="*60)
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()