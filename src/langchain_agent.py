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
                 api_key: str=None,
                 base_url: str = None,
                 model: str = None,
                 temperature: float = None):
        # 加载.env文件
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        # 从环境变量读取配置，参数优先级：显式参数 > 环境变量 > 默认值
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if base_url is None:
            base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        if model is None:
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        if temperature is None:
            temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.1"))
        
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
                "stop": stop,
                "max_tokens": 8192  # 设置API允许的最大token限制，避免内容被截断
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
            "stop": stop,
            "max_tokens": 8192  # 设置API允许的最大token限制，避免内容被截断
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
        self.use_real_llm = False
        self.api_key = ''
        
        # 初始化各个模块
        self._initialize_modules()
        
        # 分析数据集结构
        self.analyze_dataset()
        
        # 初始化向量库，用于存储和检索数据分析框架API
        self._initialize_vector_db()
        

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
        api_docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  'api')
        
        # 定义文档元数据映射
        doc_metadata = {
            '【案例】母婴市场销售情况分析.pdf': {"category": "市场销售数据分析框架", "api_name": "数据分析框架"},
            '北京高档酒店价格影响因素分析.pdf': {"category": "价格预测数据分析框架", "api_name": "数据分析框架"},
            '健身平台会员分析.pdf': {"category": "平台会员分析框架", "api_name": "数据分析框架"},
            '评论数据产品口碑分析.pdf': {"category": "产品口碑分析框架", "api_name": "数据分析框架"},
            '商品库存分析.pdf': {"category": "商品库存分析框架", "api_name": "数据分析框架"},
            '信用卡用户画像分析.pdf': {"category": "信用卡用户画像分析框架", "api_name": "数据分析框架"},
            '用python对微信好友进行分析 上午10.19.25.pdf': {"category": "微信好友分析框架", "api_name": "数据分析框架"},
            '员工流失建模与预测实例.pdf': {"category": "员工流失预测分析框架", "api_name": "数据分析框架"},
            '支付宝营销策略效果分析.pdf': {"category": "营销策略分析框架", "api_name": "数据分析框架"},
            'Airbnb产品数据分析.pdf': {"category": "电子产品分析框架", "api_name": "数据分析框架"},
            'python数据分析告诉你，为什么你的外卖总是这么慢.pdf': {"category": "外卖分析框架", "api_name": "数据分析框架"},

        }
        
        # 读取所有API文档文件
        for filename, metadata in doc_metadata.items():
            file_path = os.path.join(api_docs_dir, filename)
            if os.path.exists(file_path):
                try:
                    if filename.endswith('.pdf'):
                        # 使用PyPDF2读取PDF文件
                        from PyPDF2 import PdfReader
                        reader = PdfReader(file_path)
                        content = ""
                        for page in reader.pages:
                            content += page.extract_text() or ""
                    else:
                        # 读取普通文本文件
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    
                    # 创建初始文档对象
                    document = Document(page_content=content, metadata=metadata)
                    # 将文档分割成小块
                    doc_chunks = text_splitter.split_documents([document])
                    # 添加到初始文档列表
                    initial_documents.extend(doc_chunks)
                    print(f"已加载并分割文档: {filename} (共 {len(doc_chunks)} 个块)")
                except Exception as e:
                    print(f"警告: 读取文档时出错 {filename}: {str(e)}")
            else:
                print(f"警告: 文档文件不存在: {file_path}")
        
        # 创建内存向量库（不保存到磁盘）
        if initial_documents:
            self.vector_db = FAISS.from_documents(initial_documents, self.embeddings)
            print(f"向量库创建成功（内存模式），共加载 {len(initial_documents)} 个文档块")
        else:
            # 如果没有文档，创建一个空的向量库
            from langchain_community.docstore.in_memory import InMemoryDocstore
            from langchain_community.vectorstores.utils import DistanceStrategy
            import faiss
            
            # 创建一个空的FAISS索引
            index = faiss.IndexFlatL2(768)  # 假设使用768维的嵌入向量
            self.vector_db = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={},
                distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
            )
            print("向量库创建成功（内存模式），但未加载任何文档块")
        

    
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
        self.preprocessor.write_data_preprocessing_report()
        
        print("=== 数据预处理完成 ===")
        return "数据预处理已完成，预处理后的数据已保存到: " + self.config.PROCESSED_DATA_FILE
    
    def run_feature_engineering(self, target_col=None):
        """运行特征工程"""
        print("=== 开始特征工程 ===")
        self.feature_engineer.load_processed_data()
        self.feature_engineer.add_time_based_features()
        self.feature_engineer.encode_categorical_features(method='onehot')
        self.feature_engineer.scale_numerical_features(method='standard')
        self.feature_engineer.select_features()
        self.feature_engineer.save_features_data()
        self.feature_engineer.write_feature_engineering_report()
        
        print("=== 特征工程完成 ===")
        return "特征工程已完成，特征工程后的数据已保存到: " + self.config.FEATURES_DATA_FILE
    
    def run_visualization(self):
        """运行数据可视化"""
        print("=== 开始数据可视化 ===")
        
        import types
        
        # 定义一个替换函数，用于修改plot_from_suggestions方法
        def patched_plot_from_suggestions(self_visualizer):
            """修补后的plot_from_suggestions方法，跳过不存在的可视化方法"""
            if self_visualizer.visualization_suggestions is None:
                print("未获取到可视化建议，使用默认可视化流程")
                return
            
            print(f"\n根据可视化建议生成图表...")
            
            # 为不同图表类型分配处理函数，只保留存在的方法
            chart_handlers = {
                'histogram': self_visualizer._plot_histogram,
                'scatter': self_visualizer._plot_scatter,
                'boxplot': self_visualizer._plot_boxplot,
                'bar': self_visualizer._plot_bar,
                'line': self_visualizer._plot_line
                # 移除不存在的方法引用: 'heatmap', 'pairplot', 'feature_importance'
            }
            
            # 跟踪已生成的图表，避免重复
            self_visualizer.generated_charts = set()
            
            for suggestion in self_visualizer.visualization_suggestions:
                try:
                    # 只生成推荐的图表
                    if not suggestion.get('recommended', True):
                        continue
                    
                    chart_type = suggestion.get('chart_type')
                    title = suggestion.get('title')
                    x_axis = suggestion.get('x_axis')
                    y_axis = suggestion.get('y_axis')
                    description = suggestion.get('description')
                    
                    # 检查必要参数
                    if not chart_type or not title:
                        continue
                    
                    # 检查列是否存在
                    if x_axis and x_axis not in self_visualizer.data.columns:
                        print(f"跳过{title}: X轴字段{x_axis}不存在")
                        continue
                    if y_axis and y_axis not in self_visualizer.data.columns:
                        print(f"跳过{title}: Y轴字段{y_axis}不存在")
                        continue
                    
                    # 生成图表唯一标识
                    chart_id = f"{chart_type}_{x_axis or 'none'}_{y_axis or 'none'}"
                    if chart_id in self_visualizer.generated_charts:
                        continue
                    self_visualizer.generated_charts.add(chart_id)
                    
                    print(f"  生成图表: {title}")
                    
                    # 根据图表类型调用相应的处理函数
                    if chart_type in chart_handlers:
                        if chart_type in ['histogram', 'scatter', 'boxplot', 'bar', 'line']:
                            chart_handlers[chart_type](x_axis, y_axis, title, description)
                        else:
                            chart_handlers[chart_type]()
                    else:
                        print(f"  不支持的图表类型: {chart_type}")
                        
                except Exception as e:
                    print(f"  生成图表时出错 {suggestion.get('title', '未知')}: {e}")
                    continue
        
        # 替换visualizer的plot_from_suggestions方法
        self.visualizer.plot_from_suggestions = types.MethodType(patched_plot_from_suggestions, self.visualizer)
        
        # 执行可视化流程
  
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
       
        print("=== 开始使用自定义prompt生成综合报告 ===")
        
        report_file = self.config.ANALYSIS_REPORT
        
        # 定义API文档目录
        api_docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'api_docs')
        os.makedirs(api_docs_dir, exist_ok=True)
        
        # 初始化API文档内容
        api_docs_content = ""
        
        # 定义报告文件优先级和名称映射
        report_files = [
            ('数据预处理报告', 'data_preprocessing.md'),
            ('特征工程报告', 'feature_engineering.md'),
            ('数据可视化报告', 'data_visualization.md'),
            ('回归分析报告', 'regression_analysis.md'),
            ('场景分析报告', 'scenario_analysis_report.md')
        ]
        
        # 读取每个报告文件的内容
        for report_name, filename in report_files:
            file_path = os.path.join(api_docs_dir, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            api_docs_content += f"\n\n### {report_name}\n\n"
                            api_docs_content += content
                except Exception as e:
                    print(f"读取{report_name}时出错: {e}")
        
        # 获取生成的可视化图片列表
        visualizations_dir = "results/visualizations"
        visualization_files = []
        if os.path.exists(visualizations_dir):
            visualization_files = [f for f in os.listdir(visualizations_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.svg'))]
        
        # 构建可视化图片信息字符串
        visualization_info = ""
        if visualization_files:
            visualization_info = "\n## 可用可视化图片\n"
            visualization_info += "以下是已生成的可视化图片，可以在报告中适当位置引用：\n"
            for img_file in visualization_files:
                # 提取图片描述（从文件名中生成）
                img_desc = img_file.replace('_', ' ').replace('.png', '').replace('.jpg', '').replace('.jpeg', '').replace('.svg', '')
                visualization_info += f"- 图片路径：{os.path.join(visualizations_dir, img_file)}，描述：{img_desc}\n"
        
        # 获取最相似的数据分析框架API作为辅助提示词
        framework_query = "请提供完整的数据分析框架API，用于生成专业的数据分析报告"
        framework_info = self.retrieve_rag_info(framework_query) if hasattr(self, 'retrieve_rag_info') and self.vector_db else ""
        
        # 构建第一部分prompt：框架API、报告结构和基本要求
        prompt_part1 = f"""请根据以下推荐的数据分析框架API和报告结构要求，生成一份专业综合数据分析报告的基础框架和整体规划：

## 推荐的数据分析框架API

{framework_info}


## 报告结构要求
1. **摘要**：简明扼要地概述分析目的、主要发现和核心建议（1-2页）
2. **数据基础**：介绍数据来源、结构和主要特征
3. **分析方法**：概述所采用的分析方法和技术路线
4. **五大分析过程详解**：
   - 数据预处理
   - 特征工程
   - 数据可视化
   - 实证分析
   - 预案性分析
5. **关键洞察**：高亮显示从所有分析中得出的最重要发现（使用加粗或表格形式）
6. **建议与行动方案**：基于分析结果提供具体、可执行的建议，并按优先级排序
7. **局限性与未来工作**：讨论分析的局限性和潜在的改进方向
8. **结论**：总结分析的主要价值和贡献
9. **补充分析**：可以在报告中添加一些原理和公式，以帮助读者更好地理解分析过程

## 报告撰写基本要求
1. 语言专业但简洁明了，避免冗余和重复
2. 突出数据驱动的洞察和发现，使用具体数据支持结论
3. 建议要具体、可操作，并说明预期影响
4. 必须包含实际分析结果中的具体数据，不得用占位符替代
5. 使用清晰的标题层级和列表结构，确保报告易于导航
6. 重点内容可使用加粗或列表形式突出显示
7. 确保报告逻辑连贯，从数据到洞察再到建议形成完整闭环
8. 每个分点的字数不能太少，至少要500字左右

请为这份专业的综合数据分析报告生成详细的大纲和各部分的初步内容框架，包括报告的整体结构、各章节的主要内容要点和撰写思路。"""
        
        # 构建第二部分prompt：详细分析内容、可视化和具体要求
        prompt_part2 = f"""现在，请根据以下详细的数据分析结果、可用的可视化图片和五大分析过程的具体要求，完成刚才规划的综合数据分析报告的完整内容：

## 各阶段详细分析报告

以下是数据分析各阶段的详细报告内容：

{api_docs_content}
{visualization_info}

## 可视化图片详细要求
1. 在报告中适当位置插入相关可视化图片，使用Markdown图片格式：![图片描述](image_path)
2. **所有图片描述必须使用 <center> 标签包裹，确保在PDF中居中显示，格式为：![图片描述](image_path)<center>图1：图片描述</center>
3. **必须使用实际生成的图片**，不要使用占位符
4. 从可用可视化图片列表中选择合适的图片插入到报告的对应部分
5. **数据预处理**部分：可包含数据分布、缺失值可视化、异常值检测图表等
6. **特征工程**部分：可包含特征重要性排序、特征相关性图表等
7. **数据可视化**部分：可包含数值型特征分布、相关性热图、时间序列分析、分类特征比较、特征散点图等
8. **实证分析**部分：可包含模型性能对比图、特征系数可视化、残差分析图等
9. **预案性分析**部分：可包含场景模拟结果对比图、敏感性分析图表等
10. 每个主要分析部分至少包含1-2张相关可视化图片
11. 图片应具有清晰的标题和说明，解释其展示的内容和洞察

## 序号列表格式要求
- 序号列表的每个项目之间必须空一行，不要写到一行里
- 例如：
  ```
  1. 第一点内容
  
  2. 第二点内容
  
  3. 第三点内容
  ```

## 五大分析过程详细要求

### 1. 数据预处理
- 描述数据加载和基本信息
- 说明缺失值处理方法（删除、填充等）
- 描述异常值检测和处理方法
- 说明数据类型转换和标准化处理
- 介绍添加的衍生特征及其意义

### 2. 特征工程
- 描述分类特征的编码方法（独热编码、标签编码等）
- 说明数值特征的缩放方法（标准化、归一化等）
- 介绍特征选择方法（递归特征消除、选择K个最好特征等）
- 说明特征降维方法（如PCA）
- 描述最终选择的特征集及其理由

### 3. 数据可视化
- 展示数值型特征的分布情况
- 分析特征间的相关性
- 展示分类特征的分布情况
- 分析时间序列特征的趋势（如果有）
- 展示关键特征之间的关系

### 4. 实证分析
- 描述采用的模型（线性回归、岭回归、Lasso、随机森林、梯度提升树、LSTM等）
- 展示模型性能比较结果（如R²、MSE、MAE等）
- 分析特征重要性和模型解释
- 说明模型选择的理由

### 5. 预案性分析
- 描述场景分析的方法和假设
- 展示不同场景下的模拟结果
- 分析关键参数的敏感性
- 提供基于场景分析的洞察

请使用Markdown格式生成一份结构完整、内容丰富、包含五大分析过程的综合数据分析报告，确保使用实际的可视化图片路径，并在报告中适当位置插入这些图片。"""
        
        # 构建第三部分prompt：进一步完善报告内容
        prompt_part3 = f"""请根据以下已经生成的报告内容，进一步完善和扩展报告，使其更加详细、专业和全面：

{prompt_part2}

## 完善报告的要求
1. 深入挖掘分析结果中的关键洞察和业务价值
2. 补充更多的数据支持和详细分析
3. 丰富报告的理论基础和方法论解释
4. 扩展建议与行动方案，提供更具体的实施步骤和预期效果
5. 增强报告的可读性和专业度，确保内容更加充实和有深度
6. 确保所有部分都符合报告撰写的详细要求和格式规范

随着第二的报告继续完善"""
        
        # 生成报告内容
        if hasattr(self, 'use_real_llm') and self.use_real_llm and hasattr(self, 'api_key') and self.api_key:
            # 使用自定义的OpenAIClientLLM类直接调用DeepSeek API
            llm = OpenAIClientLLM(
                api_key=self.api_key
                # base_url和model将从.env文件加载
            )
            # 分三次调用API，逐步完善报告内容
            print("=== 正在生成报告框架... ===")
            report_framework = llm._call(prompt_part1)
            print("=== 正在生成完整报告内容... ===")
            initial_report = llm._call(prompt_part2 + f"\n\n以下是之前生成的报告框架作为参考：\n{report_framework}")
            print("=== 正在进一步完善报告... ===")
            report_content = llm._call(prompt_part3 + f"\n\n以下是已经生成的报告内容作为参考：\n{initial_report}")
        else:
            # 使用API文档内容作为报告
            report_content = f"# 数据分析报告\n\n" + api_docs_content
        
        # 将报告内容保存到文件
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"=== 报告已保存至: {report_file} ===")
            return f"报告已生成并保存至: {report_file}"
        except Exception as e:
            print(f"=== 报告保存失败: {e} ===")
            return f"报告生成失败: {e}"
 
    def create_agent_executor(self, use_real_llm=False, api_key='DEEPSEEK_API_KEY'):
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
            # 使用自定义的OpenAIClientLLM类直接调用API，配置从.env文件加载
            llm = OpenAIClientLLM(
                api_key=api_key
                # base_url和model将从.env文件加载
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
        # 加载.env文件
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        # 创建Agent实例
        agent = LangChainDataAnalysisAgent()
        
        # 从环境变量加载配置
        use_real_llm = os.getenv("USE_REAL_LLM", "True").lower() == "true"
        api_key = os.getenv("DEEPSEEK_API_KEY")

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