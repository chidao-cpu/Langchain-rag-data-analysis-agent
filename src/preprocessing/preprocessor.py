import pandas as pd
import numpy as np
from src.config import Config
import os

class DataPreprocessor:
    def __init__(self):
        self.config = Config()
        self.data = None
        
    def load_data(self, file_path=None):
        """加载原始数据"""
        file_path = file_path or self.config.RAW_DATA_FILE
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"原始数据文件不存在: {file_path}")
        
        self.data = pd.read_csv(file_path)
        print(f"成功加载数据: {file_path}")
        print(f"数据形状: {self.data.shape}")
        print(f"列名: {list(self.data.columns)}")
        print(f"\n数据前5行:\n{self.data.head()}")
        print(f"\n数据基本信息:\n")
        self.data.info()
        return self.data
    
    def handle_missing_values(self):
        """处理缺失值"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 检查缺失值情况
        missing_values = self.data.isnull().sum()
        print(f"\n缺失值情况:\n{missing_values}")
        
        # 删除缺失值比例超过阈值的列
        missing_ratio = self.data.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > self.config.MISSING_VALUE_THRESHOLD].index
        if len(cols_to_drop) > 0:
            print(f"\n删除缺失值比例超过{self.config.MISSING_VALUE_THRESHOLD}的列: {list(cols_to_drop)}")
            self.data = self.data.drop(cols_to_drop, axis=1)
        
        # 对数值型列填充均值
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        
        # 对类别型列填充众数
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        
        print(f"\n处理后缺失值情况:\n{self.data.isnull().sum()}")
        return self.data
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """检测异常值"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        method = method or self.config.OUTLIER_METHOD
        threshold = threshold or self.config.OUTLIER_THRESHOLD
        
        outliers_info = {}
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers = z_scores > threshold
            else:
                raise ValueError(f"不支持的异常值检测方法: {method}")
            
            if method == 'iqr':
                outliers = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            
            outliers_info[col] = {
                'count': outliers.sum(),
                'percentage': (outliers.sum() / len(self.data)) * 100,
                'outliers': self.data[outliers][col].tolist()
            }
            
            print(f"\n列 {col} 的异常值情况:")
            print(f"  异常值数量: {outliers_info[col]['count']}")
            print(f"  异常值比例: {outliers_info[col]['percentage']:.2f}%")
            if method == 'iqr':
                print(f"  IQR范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
        return outliers_info
    
    def remove_outliers(self, method='iqr', threshold=1.5):
        """移除异常值"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        method = method or self.config.OUTLIER_METHOD
        threshold = threshold or self.config.OUTLIER_THRESHOLD
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        original_shape = self.data.shape
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                self.data = self.data[z_scores <= threshold]
            else:
                raise ValueError(f"不支持的异常值检测方法: {method}")
        
        print(f"\n移除异常值后数据形状: {self.data.shape}")
        print(f"移除的行数: {original_shape[0] - self.data.shape[0]}")
        return self.data
    
    def convert_data_types(self):
        """自动转换数据类型"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print(f"\n开始自动转换数据类型...")
        
        # 自动检测并转换日期列
        for col in self.data.columns:
            if col.lower() in ['date', 'time', 'datetime', 'timestamp', '销售日期', '日期', '时间']:
                try:
                    self.data[col] = pd.to_datetime(self.data[col])
                    print(f"  转换列 {col} 为日期时间类型")
                except:
                    # 如果无法转换为日期时间类型，保持原样
                    pass
            else:
                # 尝试转换其他可能包含日期的列
                try:
                    # 先检查字符串长度是否合理（日期格式通常有一定长度）
                    if self.data[col].dtype == 'object' and len(self.data[col].dropna()) > 0:
                        sample_value = self.data[col].dropna().iloc[0]
                        if isinstance(sample_value, str) and (8 <= len(sample_value) <= 20):
                            self.data[col] = pd.to_datetime(self.data[col])
                            print(f"  转换列 {col} 为日期时间类型")
                except:
                    # 如果无法转换为日期时间类型，保持原样
                    pass
        
        # 自动检测并转换分类列
        # 对于object类型的列，如果唯一值数量少于数据行数的10%且少于50个，转换为category类型
        for col in self.data.select_dtypes(include=['object']).columns:
            unique_ratio = len(self.data[col].unique()) / len(self.data)
            if unique_ratio < 0.1 and len(self.data[col].unique()) < 50:
                self.data[col] = self.data[col].astype('category')
                print(f"  转换列 {col} 为分类类型")
        
        print(f"\n转换后的数据类型:\n")
        self.data.info()
        return self.data
    
    def _collect_data_info(self):
        """收集数据信息作为API请求的辅助prompt"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 获取基本信息
        data_shape = self.data.shape
        columns = list(self.data.columns)
        data_types = self.data.dtypes.astype(str).to_dict()
        
        # 获取数值型列的基本统计信息
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_stats = {} if not numeric_cols else self.data[numeric_cols].describe().round(2).to_dict()
        
        # 获取类别型列的基本信息
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_info = {}
        for col in categorical_cols:
            unique_count = len(self.data[col].unique())
            unique_values = self.data[col].unique()[:5].tolist()  # 只显示前5个唯一值
            categorical_info[col] = {
                'unique_count': unique_count,
                'sample_values': unique_values
            }
        
        # 构建数据信息字符串
        data_info = f"数据集信息：\n"
        data_info += f"- 数据形状: {data_shape[0]}行 × {data_shape[1]}列\n"
        data_info += f"- 所有列名: {columns}\n"
        data_info += f"- 数据类型: {data_types}\n"
        
        if numeric_cols:
            data_info += f"\n数值型列：\n"
            data_info += f"- 列名: {numeric_cols}\n"
            data_info += f"- 统计信息: {numeric_stats}\n"
        
        if categorical_cols:
            data_info += f"\n类别型列：\n"
            for col, info in categorical_info.items():
                data_info += f"- {col}: {info['unique_count']}个唯一值，样本值: {info['sample_values']}\n"
        
        return data_info
    
    def add_derived_columns(self):
        """根据数据集中的列自动添加衍生特征，使用DeepSeek API获取建议"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print(f"\n开始自动添加衍生特征...")
        
        # 收集数据信息作为辅助prompt
        data_info = self._collect_data_info()
        
        # 初始化DeepSeek API客户端（动态导入以避免循环导入）
        from src.langchain_agent import OpenAIClientLLM
        deepseek_client = OpenAIClientLLM()
        
        # 构建prompt
        prompt = f"你是一个数据分析师，请根据以下数据集信息，建议可以添加的衍生列：\n\n"
        prompt += data_info
        prompt += f"\n请以JSON格式输出建议的衍生列，每个衍生列应包含：\n"
        prompt += f"1. column_name: 衍生列的名称\n"
        prompt += f"2. calculation: 衍生列的计算公式（使用Python pandas语法）\n"
        prompt += f"3. description: 衍生列的描述\n"
        prompt += f"\n例如：\n"
        prompt += f"[{{\"column_name\": \"销售额\", \"calculation\": \"data['价格'] * data['销量']\", \"description\": \"产品销售额\"}}]\n"
        prompt += f"\n请只输出JSON格式的结果，不要添加任何其他文字或解释。\n"
    
        # 调用DeepSeek API获取建议
        print("  正在调用DeepSeek API获取衍生列建议...")
        response = deepseek_client.invoke(prompt)
        
        # 解析API响应
        import json
        derived_columns = json.loads(response)
        
        # 根据建议添加衍生列
        added_columns = 0
        for col in derived_columns:
            try:
                column_name = col['column_name']
                calculation = col['calculation']
                description = col['description']
                
                # 使用eval计算衍生列，这里需要注意安全问题，但在内部使用是可以接受的
                self.data[column_name] = eval(calculation.replace('data', 'self.data'))
                print(f"  生成{column_name}列")
                added_columns += 1
            except Exception as e:
                print(f"  生成衍生列失败：{str(e)}")
                continue
        

        
      
        
        # 从所有日期时间列提取时间特征（这部分无论API调用是否成功都会执行）
        datetime_cols = self.data.select_dtypes(include=['datetime64']).columns
        for date_col in datetime_cols:
            print(f"  从 {date_col} 提取时间特征")
            self.data[f'{date_col}_年份'] = self.data[date_col].dt.year
            self.data[f'{date_col}_月份'] = self.data[date_col].dt.month
            self.data[f'{date_col}_日'] = self.data[date_col].dt.day
            self.data[f'{date_col}_星期'] = self.data[date_col].dt.dayofweek
            self.data[f'{date_col}_星期名称'] = self.data[date_col].dt.day_name()
        
        return self.data
    
    def save_processed_data(self, file_path=None):
        """保存处理后的数据"""
        if self.data is None:
            raise ValueError("请先加载和处理数据")
        
        file_path = file_path or self.config.PROCESSED_DATA_FILE
        self.data.to_csv(file_path, index=False)
        print(f"\n处理后的数据已保存到: {file_path}")
        return file_path
    
    def write_data_preprocessing_report(self):
        """将数据预处理结果写入到文档"""
        if self.data is None:
            raise ValueError("请先加载和处理数据")
        
        # 获取数据信息
        data_info = self._collect_data_info()
        
        # 构建报告内容
        report_content = "\n## 数据预处理结果\n\n"
        report_content += "### 处理后的数据信息\n\n"
        report_content += "```\n"
        report_content += data_info
        report_content += "\n```\n"
        
        # 写入到data_preprocessing.md文件
        report_path = "docs/api_docs/data_preprocessing.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n数据预处理结果已写入到: {report_path}")
        return report_path
    
    def run_pipeline(self):
        """运行完整的预处理流程"""
        print("=== 开始数据预处理流程 ===")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 处理缺失值
        self.handle_missing_values()
        
        # 3. 转换数据类型
        self.convert_data_types()
        
        # 4. 检测异常值
        self.detect_outliers()
        
        # 5. 移除异常值
        self.remove_outliers()
        
        # 6. 添加衍生列
        self.add_derived_columns()
        
        # 7. 保存处理后的数据
        self.save_processed_data()
        
        # 8. 写入预处理报告
        self.write_data_preprocessing_report()
        
        print("\n=== 数据预处理流程完成 ===")
        return self.data

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_pipeline()
