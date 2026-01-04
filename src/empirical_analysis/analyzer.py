import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from src.config import Config
import os
import pickle
import warnings

# LSTM模型相关依赖
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
KERAS_AVAILABLE = True


warnings.filterwarnings('ignore')

class EmpiricalAnalyzer:
    def __init__(self):
        self.config = Config()
        self.data = None
        self.features_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.hypothesis_results = {}
        
    def load_data(self):
        """加载数据"""
        # 加载预处理后的数据
        if os.path.exists(self.config.PROCESSED_DATA_FILE):
            self.data = pd.read_csv(self.config.PROCESSED_DATA_FILE)
            if '销售日期' in self.data.columns:
                self.data['销售日期'] = pd.to_datetime(self.data['销售日期'])
            print(f"成功加载预处理后的数据: {self.config.PROCESSED_DATA_FILE}")
        else:
            raise FileNotFoundError(f"预处理后的数据文件不存在: {self.config.PROCESSED_DATA_FILE}")
        
        # 加载特征工程后的数据
        if os.path.exists(self.config.FEATURES_DATA_FILE):
            self.features_data = pd.read_csv(self.config.FEATURES_DATA_FILE)
            print(f"成功加载特征工程后的数据: {self.config.FEATURES_DATA_FILE}")
        else:
            raise FileNotFoundError(f"特征工程后的数据文件不存在: {self.config.FEATURES_DATA_FILE}")
        
        # 自动识别目标列
        self.identify_target_column()
        
        return self.data, self.features_data
    
    def identify_target_column(self):
        """自动识别目标列"""
        # 优先使用配置文件中的目标列
        if self.config.TARGET_COL is not None:
            if self.config.TARGET_COL in self.data.columns:
                self.target_col = self.config.TARGET_COL
                print(f"使用配置文件中的目标列: {self.target_col}")
                return
            elif self.features_data is not None and self.config.TARGET_COL in self.features_data.columns:
                self.target_col = self.config.TARGET_COL
                print(f"使用配置文件中的目标列: {self.target_col}")
                return
        
        # 如果有特征工程后的数据，从其中识别目标列
        if self.features_data is not None:
            # 获取特征工程后数据中的所有数值型列
            features_numeric_cols = self.features_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # 自动识别目标列（优先考虑包含'利润'、'销售额'、'销量'、'收入'等关键词的列）
            target_keywords = ['利润', '销售额', '销量', '收入', 'price', 'sales', 'revenue', 'profit']
            
            # 查找包含目标关键词的列
            for col in features_numeric_cols:
                for keyword in target_keywords:
                    if keyword in col.lower():
                        self.target_col = col
                        print(f"从特征工程后的数据中自动识别目标列为: {self.target_col}")
                        return
            
            # 如果没有找到，使用最后一个数值型列
            if features_numeric_cols:
                self.target_col = features_numeric_cols[-1]
                print(f"未找到明显的目标列，使用特征工程后数据中的最后一个数值型列作为目标列: {self.target_col}")
                return
        
        # 如果没有特征工程后的数据，或在其中没有找到目标列，从原始数据中识别
        # 自动识别目标列（优先考虑包含'利润'、'销售额'、'销量'、'收入'等关键词的列）
        target_keywords = ['利润', '销售额', '销量', '收入', 'price', 'sales', 'revenue', 'profit']
        
        # 获取所有数值型列
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 查找包含目标关键词的列
        for col in numerical_cols:
            for keyword in target_keywords:
                if keyword in col.lower():
                    self.target_col = col
                    print(f"从原始数据中自动识别目标列为: {self.target_col}")
                    return
        
        # 如果没有找到，使用最后一个数值型列
        if numerical_cols:
            self.target_col = numerical_cols[-1]
            print(f"未找到明显的目标列，使用原始数据中的最后一个数值型列作为目标列: {self.target_col}")
        else:
            self.target_col = None
            print("警告: 未找到数值型列作为目标列")
    
    def descriptive_statistics(self):
        """描述性统计分析"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print("=== 描述性统计分析 ===")
        
        # 自动获取所有数值型列
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        print("\n数值型特征统计:")
        stats_df = self.data[numerical_cols].describe()
        print(stats_df)
        
        # 保存统计结果
        stats_file = os.path.join(self.config.RESULTS_DIR, 'descriptive_statistics.csv')
        stats_df.to_csv(stats_file)
        print(f"\n描述性统计结果已保存到: {stats_file}")
        
        return stats_df
    
    def correlation_analysis(self):
        """相关性分析"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print("\n=== 相关性分析 ===")
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = self.data[numerical_cols].corr()
        
        print("\n相关性矩阵:")
        print(corr_matrix.round(2))
        
        # 找出与目标列相关性最高的特征
        if hasattr(self, 'target_col') and self.target_col in corr_matrix.columns:
            target_corr = corr_matrix[self.target_col].sort_values(ascending=False)
            print(f"\n与{self.target_col}相关性排序:")
            print(target_corr.round(2))
        
        # 保存相关性分析结果
        corr_file = os.path.join(self.config.RESULTS_DIR, 'correlation_matrix.csv')
        corr_matrix.to_csv(corr_file)
        print(f"\n相关性分析结果已保存到: {corr_file}")
        
        return corr_matrix
    
    def _collect_data_info(self):
        """收集数据信息，用于构建DeepSeek API的prompt"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        data_info = f"数据集基本信息：\n"
        data_info += f"- 数据形状：{self.data.shape}\n"
        data_info += f"- 列名：{list(self.data.columns)}\n"
        
        # 获取数值型列信息
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            data_info += f"\n数值型列：{numeric_cols}\n"
            
        # 获取类别型列信息
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            data_info += f"\n类别型列：{categorical_cols}\n"
            for col in categorical_cols:
                unique_count = self.data[col].nunique()
                sample_values = self.data[col].unique()[:3]
                data_info += f"  - {col}：{unique_count}个唯一值，样本：{sample_values}\n"
        
        # 获取目标列信息
        if hasattr(self, 'target_col') and self.target_col:
            data_info += f"\n目标列：{self.target_col}\n"
        
        return data_info
    
    def hypothesis_testing(self):
        """假设检验，使用DeepSeek API获取建议"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print("\n=== 假设检验 ===")
        
        results = {}
        
        # 收集数据信息
        data_info = self._collect_data_info()
        
        # 初始化DeepSeek API客户端（动态导入以避免循环导入）
        from src.langchain_agent import OpenAIClientLLM
        deepseek_client = OpenAIClientLLM()
        
        # 构建prompt
        prompt = f"你是一个数据分析师，请根据以下数据集信息，建议可以进行的假设检验：\n\n"
        prompt += data_info
        prompt += f"\n请以JSON格式输出建议的假设检验，每个假设检验应包含：\n"
        prompt += f"1. test_name: 检验名称\n"
        prompt += f"2. test_type: 检验类型（如t检验、方差分析、卡方检验等）\n"
        prompt += f"3. description: 检验的描述\n"
        prompt += f"4. implementation: Python代码实现（使用scipy.stats和pandas）\n"
        prompt += f"5. interpretation: 如何解释检验结果\n"
        prompt += f"\n例如：\n"
        prompt += f"[{{\"test_name\": \"促销活动对销量的影响\", \"test_type\": \"t检验\", \"description\": \"检验促销活动是否对销量有显著影响\", \"implementation\": \"t_stat, p_value = stats.ttest_ind(data[data['促销活动']=='是']['销量'], data[data['促销活动']=='否']['销量'])\", \"interpretation\": \"p值小于0.05则拒绝原假设，认为促销活动对销量有显著影响\"}}]\n"
        prompt += f"\n请只输出JSON格式的结果，不要添加任何其他文字或解释。\n"
        
        try:
            # 调用DeepSeek API获取建议
            print("  正在调用DeepSeek API获取假设检验建议...")
            response = deepseek_client.invoke(prompt)
            print(f"  API响应内容: {response}")
            
            # 解析API响应
            import json
            suggested_tests = json.loads(response)
            print(f"  解析后的假设检验建议: {suggested_tests}")
            
            # 执行建议的假设检验
            test_count = 0
            for test in suggested_tests:
                try:
                    test_name = test['test_name']
                    test_type = test['test_type']
                    description = test['description']
                    implementation = test['implementation']
                    interpretation = test['interpretation']
                    
                    print(f"\n{test_count + 1}. {test_name}（{test_type}）")
                    print(f"   描述: {description}")
                    
                    # 执行检验
                    # 使用正则表达式替换变量名data为self.data，避免替换文件名中的data
                    import re
                    # 先移除read_csv语句，因为我们已经有self.data了
                    modified_impl = re.sub(r'^data\s*=\s*pd\.read_csv\s*\(.*?\)\s*$', '', implementation, flags=re.MULTILINE)
                    # 替换变量名data为self.data
                    modified_impl = re.sub(r'\bdata\b', 'self.data', modified_impl)
                    
                    # 创建执行环境，包含self变量
                    exec_env = {'self': self, 'stats': stats, 'pd': pd}
                    try:
                        exec(modified_impl, globals(), exec_env)
                    except Exception as exec_err:
                        print(f"   执行代码错误: {str(exec_err)}")
                        continue
                    
                    # 获取执行结果
                    t_stat = exec_env.get('t_stat', None)
                    f_stat = exec_env.get('f_stat', None)
                    chi2_stat = exec_env.get('chi2_stat', None)
                    p_value = exec_env.get('p_value', None)
                    corr_coef = exec_env.get('corr_coef', None)
                    
                    # 打印结果
                    result_printed = False
                    if t_stat is not None and p_value is not None:
                        print(f"   t统计量: {t_stat:.4f}, p值: {p_value:.4f}")
                        result_printed = True
                    elif f_stat is not None and p_value is not None:
                        print(f"   F统计量: {f_stat:.4f}, p值: {p_value:.4f}")
                        result_printed = True
                    elif chi2_stat is not None and p_value is not None:
                        print(f"   卡方统计量: {chi2_stat:.4f}, p值: {p_value:.4f}")
                        result_printed = True
                    elif corr_coef is not None and p_value is not None:
                        print(f"   相关系数: {corr_coef:.4f}, p值: {p_value:.4f}")
                        result_printed = True
                    elif p_value is not None:
                        print(f"   p值: {p_value:.4f}")
                        result_printed = True
                    
                    if not result_printed:
                        print(f"   未能获取有效统计结果")
                        continue
                    
                    # 解释结果
                    if p_value < self.config.SIGNIFICANCE_LEVEL:
                        print(f"   结论: 拒绝原假设，{interpretation}")
                    else:
                        print(f"   结论: 不拒绝原假设，{interpretation}")
                    
                    # 保存结果
                    result_entry = {
                        'test_type': test_type,
                        'p_value': p_value,
                        'description': description
                    }
                    if t_stat is not None:
                        result_entry['t_stat'] = t_stat
                    elif f_stat is not None:
                        result_entry['f_stat'] = f_stat
                    elif chi2_stat is not None:
                        result_entry['chi2_stat'] = chi2_stat
                    elif corr_coef is not None:
                        result_entry['corr_coef'] = corr_coef
                    
                    results[test_name] = result_entry
                    
                    test_count += 1
                except Exception as e:
                    print(f"   执行假设检验失败：{str(e)}")
                    continue
            
            # 如果API建议的检验都执行失败，使用默认逻辑
            if test_count == 0:
                print("  API建议的检验执行失败，使用默认逻辑...")
                use_default = True
            else:
                use_default = False
                
        except Exception as e:
            print(f"  调用DeepSeek API失败：{str(e)}")
            print("  使用默认逻辑进行假设检验...")
            use_default = True
        
        # 默认假设检验逻辑
        if use_default:
            # 获取所有列名
            all_cols = self.data.columns.tolist()
            
            # 1. 检查是否有促销活动相关列
            promotion_col = None
            for col in all_cols:
                if '促销' in col or 'promotion' in col.lower():
                    promotion_col = col
                    break
            
            # 检查是否有销量相关列
            sales_col = None
            for col in all_cols:
                if '销量' in col or 'sales' in col.lower():
                    sales_col = col
                    break
            
            # 如果有促销和销量列，进行t检验
            if promotion_col and sales_col:
                print(f"\n1. {promotion_col}对{sales_col}的影响（独立样本t检验）:")
                promotion_values = self.data[promotion_col].unique()
                if len(promotion_values) == 2:
                    group1 = self.data[self.data[promotion_col] == promotion_values[0]][sales_col]
                    group2 = self.data[self.data[promotion_col] == promotion_values[1]][sales_col]
                    
                    if len(group1) > 0 and len(group2) > 0:
                        t_stat, p_value = stats.ttest_ind(group1, group2)
                        print(f"   t统计量: {t_stat:.4f}, p值: {p_value:.4f}")
                        if p_value < self.config.SIGNIFICANCE_LEVEL:
                            print(f"   结论: {promotion_col}对{sales_col}有显著影响")
                        else:
                            print(f"   结论: {promotion_col}对{sales_col}没有显著影响")
                        results['promotion_impact'] = {'t_stat': t_stat, 'p_value': p_value}
            
            # 2. 检查是否有产品类别相关列
            category_col = None
            for col in all_cols:
                if '类别' in col or 'category' in col.lower():
                    category_col = col
                    break
            
            # 如果有类别列和目标列，进行方差分析
            if category_col and hasattr(self, 'target_col') and self.target_col:
                print(f"\n2. 不同{category_col}{self.target_col}的差异（单因素方差分析）:")
                categories = self.data[category_col].unique()
                category_values = [self.data[self.data[category_col] == cat][self.target_col] for cat in categories]
                
                if len(category_values) > 1:
                    f_stat, p_value = stats.f_oneway(*category_values)
                    print(f"   F统计量: {f_stat:.4f}, p值: {p_value:.4f}")
                    if p_value < self.config.SIGNIFICANCE_LEVEL:
                        print(f"   结论: 不同{category_col}的{self.target_col}存在显著差异")
                    else:
                        print(f"   结论: 不同{category_col}的{self.target_col}没有显著差异")
                    results['category_impact'] = {'f_stat': f_stat, 'p_value': p_value}
            
            # 3. 检查是否有地区相关列
            region_col = None
            for col in all_cols:
                if '地区' in col or 'region' in col.lower():
                    region_col = col
                    break
            
            # 如果有地区列和销售额列或目标列，进行方差分析
            value_col = None
            if hasattr(self, 'target_col') and self.target_col:
                value_col = self.target_col
            else:
                for col in all_cols:
                    if '销售额' in col or 'sales' in col.lower():
                        value_col = col
                        break
            
            if region_col and value_col:
                print(f"\n3. 不同{region_col}{value_col}的差异（单因素方差分析）:")
                regions = self.data[region_col].unique()
                region_values = [self.data[self.data[region_col] == reg][value_col] for reg in regions]
                
                if len(region_values) > 1:
                    f_stat, p_value = stats.f_oneway(*region_values)
                    print(f"   F统计量: {f_stat:.4f}, p值: {p_value:.4f}")
                    if p_value < self.config.SIGNIFICANCE_LEVEL:
                        print(f"   结论: 不同{region_col}的{value_col}存在显著差异")
                    else:
                        print(f"   结论: 不同{region_col}的{value_col}没有显著差异")
                    results['region_impact'] = {'f_stat': f_stat, 'p_value': p_value}
        
        self.hypothesis_results = results
        return results
    
    def _has_time_series_data(self):
        """检查数据集是否包含时序数据"""
        # 检查原始数据中是否有日期时间列
        if self.data is not None:
            for col in self.data.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'datetime' in col.lower():
                    if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                        return True
        
        # 检查特征工程后的数据中是否有时间相关特征
        if self.features_data is not None:
            time_features = ['month', 'quarter', 'day_of_week', 'year', 'week_of_year']
            for col in time_features:
                if any(col in feat.lower() for feat in self.features_data.columns):
                    return True
        
        return False
    
    def _create_sequences(self, X, y, time_steps=30):
        """将数据转换为LSTM模型所需的序列格式 (样本数, 时间步长, 特征数)"""
        X_seq, y_seq = [], []
        for i in range(len(X) - time_steps):
            X_seq.append(X.iloc[i:(i + time_steps)].values)
            y_seq.append(y.iloc[i + time_steps])
        return np.array(X_seq), np.array(y_seq)
    
    def prepare_data_for_modeling(self):
        """准备建模数据"""
        if self.features_data is None:
            raise ValueError("请先加载特征工程后的数据")
        
        # 使用动态识别的目标列
        if not hasattr(self, 'target_col') or not self.target_col:
            self.identify_target_column()
        
        target_col = self.target_col
        
        # 分离特征和目标
        # 注意：在特征工程过程中，目标列可能已经被从特征数据中移除了
        if target_col in self.features_data.columns:
            X = self.features_data.drop(target_col, axis=1)
            y = self.features_data[target_col]
        else:
            # 如果特征数据中不包含目标列，则特征数据就是所有特征
            # 从原始数据中获取目标变量
            X = self.features_data
            if target_col in self.data.columns:
                y = self.data[target_col]
            else:
                raise ValueError(f"目标列 {target_col} 不在特征数据和原始数据中")
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE
        )
        
        # 检查是否包含时序数据
        self.has_time_series = self._has_time_series_data()
        
        # 如果有时序数据且Keras可用，准备LSTM所需的序列数据
        if self.has_time_series and KERAS_AVAILABLE:
            print(f"\n检测到时序数据，准备LSTM模型所需的序列数据...")
            self.time_steps = min(30, len(self.X_train) // 10)  # 时间步长设置为训练集的1/10或最大30
            
            self.X_train_seq, self.y_train_seq = self._create_sequences(
                self.X_train, self.y_train, self.time_steps
            )
            self.X_test_seq, self.y_test_seq = self._create_sequences(
                self.X_test, self.y_test, self.time_steps
            )
            
            print(f"序列数据准备完成:")
            print(f"训练序列大小: {self.X_train_seq.shape}")
            print(f"测试序列大小: {self.X_test_seq.shape}")
        
        print(f"\n建模数据准备完成:")
        print(f"训练集大小: {self.X_train.shape[0]}")
        print(f"测试集大小: {self.X_test.shape[0]}")
        print(f"是否包含时序数据: {self.has_time_series}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_regression_models(self):
        """训练回归模型"""
        if self.X_train is None or self.X_test is None:
            raise ValueError("请先准备建模数据")
        
        print("\n=== 回归模型训练 ===")
        
        # 定义模型
        model_list = {
            '线性回归': LinearRegression(),
            '岭回归': Ridge(alpha=1.0),
            'Lasso回归': Lasso(alpha=1.0),
            '随机森林': RandomForestRegressor(n_estimators=100, random_state=self.config.RANDOM_STATE),
            '梯度提升树': GradientBoostingRegressor(n_estimators=100, random_state=self.config.RANDOM_STATE)
        }
        
        # 训练和评估模型
        results = []
        
        for model_name, model in model_list.items():
            print(f"\n训练{model_name}...")
            
            # 训练模型
            model.fit(self.X_train, self.y_train)
            self.models[model_name] = model
            
            # 预测
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # 评估指标
            train_mse = mean_squared_error(self.y_train, y_pred_train)
            test_mse = mean_squared_error(self.y_test, y_pred_test)
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            
            # 交叉验证
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # 保存结果
            results.append({
                '模型名称': model_name,
                '训练集MSE': train_mse,
                '测试集MSE': test_mse,
                '训练集MAE': train_mae,
                '测试集MAE': test_mae,
                '训练集R²': train_r2,
                '测试集R²': test_r2,
                '交叉验证R²均值': cv_mean,
                '交叉验证R²标准差': cv_std
            })
            
            print(f"{model_name}评估结果:")
            print(f"  训练集MSE: {train_mse:.4f}")
            print(f"  测试集MSE: {test_mse:.4f}")
            print(f"  训练集MAE: {train_mae:.4f}")
            print(f"  测试集MAE: {test_mae:.4f}")
            print(f"  训练集R²: {train_r2:.4f}")
            print(f"  测试集R²: {test_r2:.4f}")
            print(f"  交叉验证R²: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # 如果有时序数据且Keras可用，训练LSTM模型
        if self.has_time_series and KERAS_AVAILABLE:
            print(f"\n训练LSTM模型...")
            
            # 获取序列数据
            X_train_seq = getattr(self, 'X_train_seq', None)
            y_train_seq = getattr(self, 'y_train_seq', None)
            X_test_seq = getattr(self, 'X_test_seq', None)
            y_test_seq = getattr(self, 'y_test_seq', None)
            
            if X_train_seq is not None and y_train_seq is not None and X_test_seq is not None and y_test_seq is not None:
                # 构建LSTM模型
                n_features = X_train_seq.shape[2]  # 特征数量
                
                lstm_model = Sequential([
                    LSTM(64, input_shape=(self.time_steps, n_features), return_sequences=True),
                    Dropout(0.2),
                    LSTM(32),
                    Dense(1)
                ])
                
                # 编译模型
                lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                
                # 定义早停策略
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                # 训练模型
                history = lstm_model.fit(
                    X_train_seq, y_train_seq,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[early_stopping],
                    verbose=1
                )
                
                # 保存模型
                self.models['LSTM'] = lstm_model
                
                # 预测
                y_pred_train = lstm_model.predict(X_train_seq)
                y_pred_test = lstm_model.predict(X_test_seq)
                
                # 评估指标
                train_mse = mean_squared_error(y_train_seq, y_pred_train)
                test_mse = mean_squared_error(y_test_seq, y_pred_test)
                train_mae = mean_absolute_error(y_train_seq, y_pred_train)
                test_mae = mean_absolute_error(y_test_seq, y_pred_test)
                train_r2 = r2_score(y_train_seq, y_pred_train)
                test_r2 = r2_score(y_test_seq, y_pred_test)
                
                # 保存结果（LSTM模型不进行交叉验证）
                results.append({
                    '模型名称': 'LSTM',
                    '训练集MSE': train_mse,
                    '测试集MSE': test_mse,
                    '训练集MAE': train_mae,
                    '测试集MAE': test_mae,
                    '训练集R²': train_r2,
                    '测试集R²': test_r2,
                    '交叉验证R²均值': None,
                    '交叉验证R²标准差': None
                })
                
                print(f"LSTM评估结果:")
                print(f"  训练集MSE: {train_mse:.4f}")
                print(f"  测试集MSE: {test_mse:.4f}")
                print(f"  训练集MAE: {train_mae:.4f}")
                print(f"  测试集MAE: {test_mae:.4f}")
                print(f"  训练集R²: {train_r2:.4f}")
                print(f"  测试集R²: {test_r2:.4f}")
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='测试集R²', ascending=False)
        
        print("\n=== 模型性能比较 ===")
        print(results_df)
        
        # 保存模型训练结果
        model_results_file = os.path.join(self.config.RESULTS_DIR, 'model_training_results.csv')
        results_df.to_csv(model_results_file, index=False)
        print(f"\n模型训练结果已保存到: {model_results_file}")
        
        # 保存最佳模型
        best_model_name = results_df.iloc[0]['模型名称']
        best_model = self.models[best_model_name]
        
        # 根据模型类型选择不同的保存方式
        if best_model_name == 'LSTM' and KERAS_AVAILABLE:
            # 保存Keras模型
            best_model.save(self.config.MODEL_FILE.replace('.pkl', '.h5'))
            print(f"\n最佳模型({best_model_name})已保存到: {self.config.MODEL_FILE.replace('.pkl', '.h5')}")
        else:
            # 保存传统机器学习模型
            with open(self.config.MODEL_FILE, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"\n最佳模型({best_model_name})已保存到: {self.config.MODEL_FILE}")
        
        self.results = results_df
        return results_df
    
    def train_classification_models(self):
        """训练分类模型"""
        if self.X_train is None or self.X_test is None:
            raise ValueError("请先准备建模数据")
        
        print("\n=== 分类模型训练 ===")
        
        # 定义模型
        model_list = {
            '逻辑回归': LogisticRegression(random_state=self.config.RANDOM_STATE),
            '决策树': DecisionTreeClassifier(random_state=self.config.RANDOM_STATE),
            'K近邻': KNeighborsClassifier(),
            '支持向量机': SVC(random_state=self.config.RANDOM_STATE, probability=True),
            '随机森林': RandomForestClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE),
            '梯度提升树': GradientBoostingClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE)
        }
        
        # 训练和评估模型
        results = []
        
        for model_name, model in model_list.items():
            print(f"\n训练{model_name}...")
            
            # 训练模型
            model.fit(self.X_train, self.y_train)
            self.models[model_name] = model
            
            # 预测
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # 预测概率（用于ROC AUC）
            if hasattr(model, 'predict_proba'):
                y_pred_train_proba = model.predict_proba(self.X_train)[:, 1]
                y_pred_test_proba = model.predict_proba(self.X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_pred_train_proba = model.decision_function(self.X_train)
                y_pred_test_proba = model.decision_function(self.X_test)
            else:
                y_pred_train_proba = y_pred_train
                y_pred_test_proba = y_pred_test
            
            # 评估指标
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)
            train_precision = precision_score(self.y_train, y_pred_train, average='weighted')
            test_precision = precision_score(self.y_test, y_pred_test, average='weighted')
            train_recall = recall_score(self.y_train, y_pred_train, average='weighted')
            test_recall = recall_score(self.y_test, y_pred_test, average='weighted')
            train_f1 = f1_score(self.y_train, y_pred_train, average='weighted')
            test_f1 = f1_score(self.y_test, y_pred_test, average='weighted')
            
            # ROC AUC（仅适用于二分类）
            if len(np.unique(self.y_train)) == 2:
                train_roc_auc = roc_auc_score(self.y_train, y_pred_train_proba)
                test_roc_auc = roc_auc_score(self.y_test, y_pred_test_proba)
            else:
                train_roc_auc = None
                test_roc_auc = None
            
            # 交叉验证
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # 保存结果
            results.append({
                '模型名称': model_name,
                '训练集准确率': train_accuracy,
                '测试集准确率': test_accuracy,
                '训练集精确率': train_precision,
                '测试集精确率': test_precision,
                '训练集召回率': train_recall,
                '测试集召回率': test_recall,
                '训练集F1值': train_f1,
                '测试集F1值': test_f1,
                '训练集ROC AUC': train_roc_auc,
                '测试集ROC AUC': test_roc_auc,
                '交叉验证准确率均值': cv_mean,
                '交叉验证准确率标准差': cv_std
            })
            
            print(f"{model_name}评估结果:")
            print(f"  训练集准确率: {train_accuracy:.4f}")
            print(f"  测试集准确率: {test_accuracy:.4f}")
            print(f"  训练集精确率: {train_precision:.4f}")
            print(f"  测试集精确率: {test_precision:.4f}")
            print(f"  训练集召回率: {train_recall:.4f}")
            print(f"  测试集召回率: {test_recall:.4f}")
            print(f"  训练集F1值: {train_f1:.4f}")
            print(f"  测试集F1值: {test_f1:.4f}")
            if train_roc_auc is not None:
                print(f"  训练集ROC AUC: {train_roc_auc:.4f}")
                print(f"  测试集ROC AUC: {test_roc_auc:.4f}")
            print(f"  交叉验证准确率: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='测试集准确率', ascending=False)
        
        print("\n=== 模型性能比较 ===")
        print(results_df)
        
        # 保存模型训练结果
        model_results_file = os.path.join(self.config.RESULTS_DIR, 'model_training_results.csv')
        results_df.to_csv(model_results_file, index=False)
        print(f"\n模型训练结果已保存到: {model_results_file}")
        
        # 保存最佳模型
        best_model_name = results_df.iloc[0]['模型名称']
        best_model = self.models[best_model_name]
        
        # 根据模型类型选择不同的保存方式
        if best_model_name == 'LSTM' and KERAS_AVAILABLE:
            # 保存Keras模型
            best_model.save(self.config.MODEL_FILE.replace('.pkl', '.h5'))
            print(f"\n最佳模型({best_model_name})已保存到: {self.config.MODEL_FILE.replace('.pkl', '.h5')}")
        else:
            # 保存传统机器学习模型
            with open(self.config.MODEL_FILE, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"\n最佳模型({best_model_name})已保存到: {self.config.MODEL_FILE}")
        
        self.results = results_df
        return results_df
    
    def model_interpretation(self):
        """模型解释"""
        if not self.models:
            raise ValueError("请先训练模型")
        
        print("\n=== 模型解释 ===")
        
        # 线性模型的系数分析
        linear_models = ['线性回归', '岭回归', 'Lasso回归']
        
        for model_name in linear_models:
            if model_name in self.models:
                model = self.models[model_name]
                print(f"\n{model_name}系数分析:")
                
                if hasattr(model, 'coef_'):
                    coefficients = pd.DataFrame({
                        '特征': self.X_train.columns,
                        '系数': model.coef_,
                        '绝对值': np.abs(model.coef_)
                    })
                    coefficients = coefficients.sort_values(by='绝对值', ascending=False)
                    print(coefficients.round(4))
        
        # 树模型的特征重要性
        tree_models = ['随机森林', '梯度提升树']
        
        for model_name in tree_models:
            if model_name in self.models:
                model = self.models[model_name]
                print(f"\n{model_name}特征重要性:")
                
                if hasattr(model, 'feature_importances_'):
                    importances = pd.DataFrame({
                        '特征': self.X_train.columns,
                        '重要性': model.feature_importances_
                    })
                    importances = importances.sort_values(by='重要性', ascending=False)
                    print(importances.round(4))
        
        return coefficients if '线性回归' in self.models else importances
    
    def write_regression_analysis_report(self):
        """将回归分析结果写入到regression_analysis.md文档"""
        if self.results is None:
            raise ValueError("请先运行实证分析")
        
        # 构建报告内容
        report_content = "\n## 回归分析结果\n\n"
        
        # 添加假设检验结果
        if self.hypothesis_results:
            report_content += "### 假设检验结果\n\n"
            for test_name, result in self.hypothesis_results.items():
                report_content += f"#### {test_name}\n\n"
                report_content += "```\n"
                for metric, value in result.items():
                    if value is not None:
                        if isinstance(value, (int, float, np.number)):
                            report_content += f"{metric}: {value:.4f}\n"
                        else:
                            report_content += f"{metric}: {value}\n"
                report_content += "```\n\n"
        
        # 添加模型性能比较
        report_content += "### 模型性能比较\n\n"
        report_content += "```\n"
        report_content += self.results.to_string()
        report_content += "\n```\n\n"
        
        # 添加模型解释
        report_content += "### 模型解释\n\n"
        
        # 线性模型的系数分析
        linear_models = ['线性回归', '岭回归', 'Lasso回归']
        for model_name in linear_models:
            if model_name in self.models:
                model = self.models[model_name]
                report_content += f"#### {model_name}系数分析\n\n"
                
                if hasattr(model, 'coef_'):
                    coefficients = pd.DataFrame({
                        '特征': self.X_train.columns,
                        '系数': model.coef_,
                        '绝对值': np.abs(model.coef_)
                    })
                    coefficients = coefficients.sort_values(by='绝对值', ascending=False)
                    report_content += "```\n"
                    report_content += coefficients.round(4).to_string()
                    report_content += "\n```\n\n"
        
        # 树模型的特征重要性
        tree_models = ['随机森林', '梯度提升树']
        for model_name in tree_models:
            if model_name in self.models:
                model = self.models[model_name]
                report_content += f"#### {model_name}特征重要性\n\n"
                
                if hasattr(model, 'feature_importances_'):
                    importances = pd.DataFrame({
                        '特征': self.X_train.columns,
                        '重要性': model.feature_importances_
                    })
                    importances = importances.sort_values(by='重要性', ascending=False)
                    report_content += "```\n"
                    report_content += importances.round(4).to_string()
                    report_content += "\n```\n\n"
        
        # 写入到regression_analysis.md文件
        report_path = "docs/api_docs/regression_analysis.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n回归分析结果已写入到: {report_path}")
        return report_path
    
    def run_analysis(self):
        """运行完整的实证分析流程"""
        # 1. 加载数据
        self.load_data()
        
        # 2. 描述性统计分析
        self.descriptive_statistics()
        
        # 3. 相关性分析
        self.correlation_analysis()
        
        # 4. 假设检验
        self.hypothesis_testing()
        
        # 5. 准备建模数据
        self.prepare_data_for_modeling()
        
        # 6. 根据任务类型选择训练模型
        if self.config.TASK_TYPE == 'regression':
            self.train_regression_models()
        elif self.config.TASK_TYPE == 'classification':
            self.train_classification_models()
        
        # 7. 模型解释
        self.model_interpretation()
        
        # 8. 将回归分析结果写入文档
        self.write_regression_analysis_report()
        
        print("\n=== 实证分析流程完成 ===")
        return self.results

if __name__ == "__main__":
    analyzer = EmpiricalAnalyzer()
    analyzer.run_analysis()
