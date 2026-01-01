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
    
    def add_derived_columns(self):
        """根据数据集中的列自动添加衍生特征"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print(f"\n开始自动添加衍生特征...")
        
        # 识别价格、销量、成本等列
        price_cols = [col for col in self.data.columns if 'price' in col.lower() or '价格' in col or '单价' in col]
        quantity_cols = [col for col in self.data.columns if 'quantity' in col.lower() or '销量' in col or '数量' in col]
        cost_cols = [col for col in self.data.columns if 'cost' in col.lower() or '成本' in col or 'cost' in col.lower()]
        
        # 计算销售额（价格 * 销量）
        if price_cols and quantity_cols:
            price_col = price_cols[0]
            quantity_col = quantity_cols[0]
            self.data['销售额'] = self.data[price_col] * self.data[quantity_col]
            print(f"  生成销售额列（{price_col} * {quantity_col}）")
        
        # 计算利润（销售额 - 成本 * 销量）
        if '销售额' in self.data.columns and cost_cols and quantity_cols:
            cost_col = cost_cols[0]
            quantity_col = quantity_cols[0]
            self.data['利润'] = self.data['销售额'] - (self.data[cost_col] * self.data[quantity_col])
            print(f"  生成利润列（销售额 - {cost_col} * {quantity_col}）")
        
        # 计算利润率（利润 / 销售额 * 100）
        if '利润' in self.data.columns and '销售额' in self.data.columns:
            self.data['利润率'] = (self.data['利润'] / self.data['销售额']) * 100
            self.data['利润率'] = self.data['利润率'].round(2)
            print(f"  生成利润率列")
        
        # 从所有日期时间列提取时间特征
        datetime_cols = self.data.select_dtypes(include=['datetime64']).columns
        for date_col in datetime_cols:
            print(f"  从 {date_col} 提取时间特征")
            self.data[f'{date_col}_年份'] = self.data[date_col].dt.year
            self.data[f'{date_col}_月份'] = self.data[date_col].dt.month
            self.data[f'{date_col}_日'] = self.data[date_col].dt.day
            self.data[f'{date_col}_星期'] = self.data[date_col].dt.dayofweek
            self.data[f'{date_col}_星期名称'] = self.data[date_col].dt.day_name()
        
        print(f"\n添加衍生列后的数据前5行:\n{self.data.head()}")
        return self.data
    
    def save_processed_data(self, file_path=None):
        """保存处理后的数据"""
        if self.data is None:
            raise ValueError("请先加载和处理数据")
        
        file_path = file_path or self.config.PROCESSED_DATA_FILE
        self.data.to_csv(file_path, index=False)
        print(f"\n处理后的数据已保存到: {file_path}")
        return file_path
    
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
        
        print("\n=== 数据预处理流程完成 ===")
        return self.data

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_pipeline()
