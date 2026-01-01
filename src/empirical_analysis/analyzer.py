import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score)
from src.config import Config
import os
import pickle
import warnings

# LSTM模型相关依赖
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

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
        
        return self.data, self.features_data
    
    def descriptive_statistics(self):
        """描述性统计分析"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print("=== 描述性统计分析 ===")
        
        # 数值型特征统计
        numerical_cols = ['价格', '销量', '成本', '客户评分', '销售额', '利润', '利润率']
        numerical_cols = [col for col in numerical_cols if col in self.data.columns]
        
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
        
        # 找出与利润相关性最高的特征
        profit_corr = corr_matrix['利润'].sort_values(ascending=False)
        print("\n与利润相关性排序:")
        print(profit_corr.round(2))
        
        # 保存相关性分析结果
        corr_file = os.path.join(self.config.RESULTS_DIR, 'correlation_matrix.csv')
        corr_matrix.to_csv(corr_file)
        print(f"\n相关性分析结果已保存到: {corr_file}")
        
        return corr_matrix
    
    def hypothesis_testing(self):
        """假设检验"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print("\n=== 假设检验 ===")
        
        results = {}
        
        # 1. 促销活动对销量的影响（t检验）
        print("\n1. 促销活动对销量的影响（独立样本t检验）:")
        promotion = self.data[self.data['促销活动'] == '是']['销量']
        no_promotion = self.data[self.data['促销活动'] == '否']['销量']
        t_stat, p_value = stats.ttest_ind(promotion, no_promotion)
        print(f"t统计量: {t_stat:.4f}, p值: {p_value:.4f}")
        if p_value < self.config.SIGNIFICANCE_LEVEL:
            print("结论: 促销活动对销量有显著影响")
        else:
            print("结论: 促销活动对销量没有显著影响")
        results['promotion_sales'] = {'t_stat': t_stat, 'p_value': p_value}
        
        # 2. 不同产品类别利润的差异（方差分析）
        print("\n2. 不同产品类别利润的差异（单因素方差分析）:")
        categories = self.data['产品类别'].unique()
        category_profits = [self.data[self.data['产品类别'] == cat]['利润'] for cat in categories]
        f_stat, p_value = stats.f_oneway(*category_profits)
        print(f"F统计量: {f_stat:.4f}, p值: {p_value:.4f}")
        if p_value < self.config.SIGNIFICANCE_LEVEL:
            print("结论: 不同产品类别的利润存在显著差异")
        else:
            print("结论: 不同产品类别的利润没有显著差异")
        results['category_profit'] = {'f_stat': f_stat, 'p_value': p_value}
        
        # 3. 不同地区销售额的差异（方差分析）
        print("\n3. 不同地区销售额的差异（单因素方差分析）:")
        regions = self.data['地区'].unique()
        region_sales = [self.data[self.data['地区'] == reg]['销售额'] for reg in regions]
        f_stat, p_value = stats.f_oneway(*region_sales)
        print(f"F统计量: {f_stat:.4f}, p值: {p_value:.4f}")
        if p_value < self.config.SIGNIFICANCE_LEVEL:
            print("结论: 不同地区的销售额存在显著差异")
        else:
            print("结论: 不同地区的销售额没有显著差异")
        results['region_sales'] = {'f_stat': f_stat, 'p_value': p_value}
        
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
    
    def prepare_data_for_modeling(self, target_col='利润'):
        """准备建模数据"""
        if self.features_data is None:
            raise ValueError("请先加载特征工程后的数据")
        
        # 分离特征和目标
        X = self.features_data.drop(target_col, axis=1)
        y = self.features_data[target_col]
        
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
        
        # 6. 训练回归模型
        self.train_regression_models()
        
        # 7. 模型解释
        self.model_interpretation()
        
        print("\n=== 实证分析流程完成 ===")
        return self.results

if __name__ == "__main__":
    analyzer = EmpiricalAnalyzer()
    analyzer.run_analysis()
