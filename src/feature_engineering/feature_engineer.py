import pandas as pd
import numpy as np
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, 
                                   OneHotEncoder, LabelEncoder)
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.config import Config
import os

class FeatureEngineer:
    def __init__(self):
        self.config = Config()
        self.data = None
        self.X = None
        self.y = None
        self.categorical_cols = []
        self.numerical_cols = []
        
    def load_processed_data(self, file_path=None):
        """加载预处理后的数据"""
        file_path = file_path or self.config.PROCESSED_DATA_FILE
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"预处理后的数据文件不存在: {file_path}")
        
        self.data = pd.read_csv(file_path)
        # 转换日期列
        if '销售日期' in self.data.columns:
            self.data['销售日期'] = pd.to_datetime(self.data['销售日期'])
        
        # 识别类别型和数值型列
        self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 移除不需要的列
        self.categorical_cols = [col for col in self.categorical_cols if col not in ['产品名称']]
        
        print(f"成功加载预处理后的数据: {file_path}")
        print(f"数据形状: {self.data.shape}")
        print(f"类别型特征: {self.categorical_cols}")
        print(f"数值型特征: {self.numerical_cols}")
        return self.data
    
    def encode_categorical_features(self, method='onehot'):
        """对类别型特征进行编码"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        encoded_data = self.data.copy()
        
        for col in self.categorical_cols:
            if method == 'onehot':
                # 独热编码
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded = encoder.fit_transform(encoded_data[[col]])
                encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{cat}" for cat in encoder.categories_[0][1:]])
                encoded_data = pd.concat([encoded_data.drop(col, axis=1), encoded_df], axis=1)
            elif method == 'label':
                # 标签编码
                encoder = LabelEncoder()
                encoded_data[col] = encoder.fit_transform(encoded_data[col])
            else:
                raise ValueError(f"不支持的编码方法: {method}")
        
        self.data = encoded_data
        
        # 更新数值型特征列表（包含编码后的特征）
        self.numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\n编码后的数据形状: {self.data.shape}")
        print(f"更新后的数值型特征: {self.numerical_cols}")
        return self.data
    
    def scale_numerical_features(self, method='standard'):
        """对数值型特征进行标准化/归一化"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        scaled_data = self.data.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的缩放方法: {method}")
        
        # 只对原始数值型特征进行缩放，不包括编码后的特征
        original_numerical = [col for col in self.numerical_cols if not any(cat in col for cat in self.categorical_cols)]
        scaled_data[original_numerical] = scaler.fit_transform(scaled_data[original_numerical])
        
        self.data = scaled_data
        print(f"\n数值型特征{method}缩放完成")
        return self.data
    
    def select_features(self, target_col=None, method='rfecv', task_type='regression'):
        """特征选择，自动识别目标列"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 自动识别目标列
        if not target_col:
            print(f"\n自动识别目标列...")
            # 识别可能的目标列：数值型列，不是明显的ID或时间相关列
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            possible_target_cols = []
            
            for col in numeric_cols:
                # 排除明显不是目标的列
                if not (col.lower() in ['id', 'index', 'year', 'month', 'day', 'quarter', '日期', '时间'] or 
                        'id' in col.lower() or 'index' in col.lower() or 
                        'year' in col.lower() or 'month' in col.lower() or 'day' in col.lower()):
                    possible_target_cols.append(col)
            
            if possible_target_cols:
                target_col = possible_target_cols[0]
                print(f"  自动选择目标列: {target_col}")
            else:
                raise ValueError("无法自动识别目标列，请手动指定")
        
        if target_col not in self.data.columns:
            raise ValueError(f"目标列{target_col}不存在于数据中")
        
        # 移除日期时间列
        data_for_selection = self.data.copy()
        datetime_cols = data_for_selection.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            data_for_selection = data_for_selection.drop(datetime_cols, axis=1)
        
        # 分离特征和目标
        self.X = data_for_selection.drop(target_col, axis=1)
        self.y = data_for_selection[target_col]
        
        # 选择模型
        if task_type == 'classification':
            model = RandomForestClassifier(random_state=self.config.RANDOM_STATE)
        else:
            model = RandomForestRegressor(random_state=self.config.RANDOM_STATE)
        
        if method == 'rfecv':
            # 使用递归特征消除和交叉验证
            selector = RFECV(estimator=model, cv=5, scoring='r2' if task_type == 'regression' else 'accuracy')
            selector.fit(self.X, self.y)
            selected_features = self.X.columns[selector.support_].tolist()
            
            print(f"\nRFECV特征选择结果:")
            print(f"选择的特征数量: {selector.n_features_}")
            print(f"选择的特征: {selected_features}")
            print(f"特征排名: {dict(zip(self.X.columns, selector.ranking_))}")
            
        elif method == 'selectkbest':
            # 使用SelectKBest
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(self.X, self.y)
            scores = selector.scores_
            
            # 按得分排序
            feature_scores = pd.DataFrame({'特征': self.X.columns, '得分': scores})
            feature_scores = feature_scores.sort_values(by='得分', ascending=False)
            
            print(f"\nSelectKBest特征选择结果:")
            print(feature_scores)
            
            # 选择得分最高的k个特征
            k = min(10, len(self.X.columns))
            selected_features = feature_scores['特征'][:k].tolist()
            print(f"\n选择前{k}个特征: {selected_features}")
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
        
        # 更新特征集
        self.X = self.X[selected_features]
        self.numerical_cols = selected_features
        
        print(f"\n特征选择后的数据形状: {self.X.shape}")
        return self.X, self.y
    
    def reduce_dimensionality(self, n_components=2):
        """使用PCA进行降维"""
        if self.X is None:
            raise ValueError("请先进行特征选择")
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(self.X)
        
        print(f"\nPCA降维结果:")
        print(f"降维后的数据形状: {X_pca.shape}")
        print(f"解释方差比例: {pca.explained_variance_ratio_}")
        print(f"累计解释方差比例: {sum(pca.explained_variance_ratio_)}")
        
        # 创建降维后的DataFrame
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        
        return X_pca_df
    
    def add_time_based_features(self):
        """自动识别并添加基于时间的特征"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print(f"\n开始自动添加基于时间的特征...")
        
        # 识别所有日期时间列
        datetime_cols = self.data.select_dtypes(include=['datetime64']).columns.tolist()
        
        if not datetime_cols:
            # 尝试识别可能是日期但未被正确识别的列
            for col in self.data.columns:
                if col.lower() in ['date', 'time', 'datetime', 'timestamp', '销售日期', '日期', '时间']:
                    try:
                        self.data[col] = pd.to_datetime(self.data[col])
                        datetime_cols.append(col)
                        print(f"  识别并转换列 {col} 为日期时间类型")
                    except:
                        pass
        
        if not datetime_cols:
            print("  未找到日期时间列，跳过添加时间特征")
            return self.data
        
        time_data = self.data.copy()
        
        # 为每个日期时间列添加时间特征
        for date_col in datetime_cols:
            print(f"  为列 {date_col} 添加时间特征")
            
            # 确保日期列是datetime类型
            time_data[date_col] = pd.to_datetime(time_data[date_col])
            
            # 添加季度
            time_data[f'{date_col}_季度'] = time_data[date_col].dt.quarter
            
            # 添加是否为周末
            time_data[f'{date_col}_是否周末'] = time_data[date_col].dt.dayofweek.isin([5, 6]).astype(int)
            
            # 添加月份名称
            time_data[f'{date_col}_月份名称'] = time_data[date_col].dt.month_name()
            
            # 添加年份和月份的组合
            time_data[f'{date_col}_年月'] = time_data[date_col].dt.strftime('%Y-%m')
            
            # 添加是否为月初/月中/月末
            time_data[f'{date_col}_月初'] = (time_data[date_col].dt.day <= 10).astype(int)
            time_data[f'{date_col}_月中'] = ((time_data[date_col].dt.day > 10) & (time_data[date_col].dt.day <= 20)).astype(int)
            time_data[f'{date_col}_月末'] = (time_data[date_col].dt.day > 20).astype(int)
        
        self.data = time_data
        # 更新类别型和数值型列
        self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\n添加基于时间的特征后的数据形状: {self.data.shape}")
        return self.data
    
    def create_features_data(self, target_col=None):
        """创建用于建模的特征数据"""
        if self.X is None or self.y is None:
            raise ValueError("请先进行特征选择")
        
        features_data = pd.concat([self.X, self.y], axis=1)
        return features_data
    
    def save_features_data(self, file_path=None):
        """保存特征工程后的数据"""
        features_data = self.create_features_data()
        file_path = file_path or self.config.FEATURES_DATA_FILE
        features_data.to_csv(file_path, index=False)
        print(f"\n特征工程后的数据已保存到: {file_path}")
        return file_path
    
    def run_pipeline(self, target_col=None):
        """运行完整的特征工程流程"""
        print("=== 开始特征工程流程 ===")
        
        # 1. 加载预处理后的数据
        self.load_processed_data()
        
        # 2. 添加基于时间的特征
        self.add_time_based_features()
        
        # 3. 对类别型特征进行编码
        self.encode_categorical_features(method='onehot')
        
        # 4. 对数值型特征进行标准化
        self.scale_numerical_features(method='standard')
        
        # 5. 特征选择
        self.select_features(target_col=target_col, method=self.config.FEATURE_SELECTION_METHOD)
        
        # 6. 保存特征工程后的数据
        self.save_features_data()
        
        print("\n=== 特征工程流程完成 ===")
        return self.X, self.y

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.run_pipeline()
