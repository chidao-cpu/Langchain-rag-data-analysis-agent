import pandas as pd
import numpy as np
import pickle
import os
import warnings
from src.config import Config

warnings.filterwarnings('ignore')

class ScenarioAnalyzer:
    def __init__(self):
        self.config = Config()
        self.data = None
        self.features_data = None
        self.model = None
        self.results = {}
    
    def load_data_and_model(self):
        """加载数据和训练好的模型"""
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
        
        # 加载训练好的模型
        if os.path.exists(self.config.MODEL_FILE):
            with open(self.config.MODEL_FILE, 'rb') as f:
                self.model = pickle.load(f)
            print(f"成功加载训练好的模型: {self.config.MODEL_FILE}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.config.MODEL_FILE}")
        
        return self.data, self.features_data, self.model
    
    def get_current_baseline(self):
        """获取当前基线数据"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        baseline = {
            '平均价格': self.data['价格'].mean(),
            '平均销量': self.data['销量'].mean(),
            '平均成本': self.data['成本'].mean(),
            '平均利润': self.data['利润'].mean(),
            '总销售额': self.data['销售额'].sum(),
            '总利润': self.data['利润'].sum(),
            '产品类别数': self.data['产品类别'].nunique(),
            '地区数': self.data['地区'].nunique()
        }
        
        print("\n=== 当前基线数据 ===")
        for key, value in baseline.items():
            if '平均' in key or '总' in key:
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        return baseline
    
    def scenario_sales_increase(self):
        """销售增长场景分析"""
        print("\n=== 销售增长场景分析 ===")
        
        # 获取当前特征数据（不包含目标列）
        target_col = '利润'
        X = self.features_data.drop(target_col, axis=1)
        
        # 获取原始销量数据的均值
        original_sales_mean = self.data['销量'].mean()
        
        results = []
        
        # 模拟不同的销售增长百分比
        for increase_pct in self.config.SCENARIO_PARAMS['sales_increase']:
            print(f"\n销售增长 {increase_pct}% 场景:")
            
            # 创建场景数据副本
            scenario_data = self.features_data.copy()
            
            # 检查'价格'是否是索引
            if scenario_data.index.name == '价格':
                # 将索引转换为普通列
                scenario_data = scenario_data.reset_index()
            
            # 增加销量特征
            if '销量' in scenario_data.columns:
                # 由于数据已经标准化，我们需要调整增加量
                # 计算原始数据的标准差和均值，将百分比增长转换为标准化后的增量
                original_sales_std = self.data['销量'].std()
                original_sales_mean = self.data['销量'].mean()
                
                # 将标准化的销量转换回原始值
                scenario_data['销量_original'] = scenario_data['销量'] * original_sales_std + original_sales_mean
                
                # 应用增长百分比
                scenario_data['销量_original'] = scenario_data['销量_original'] * (1 + increase_pct / 100)
                
                # 将调整后的销量转换回标准化值
                scenario_data['销量'] = (scenario_data['销量_original'] - original_sales_mean) / original_sales_std
                
                # 清理临时列
                scenario_data = scenario_data.drop('销量_original', axis=1)
            
            # 处理可能的销量衍生特征
            sales_related_cols = [col for col in scenario_data.columns if '销量' in col or 'sales' in col.lower()]
            for col in sales_related_cols:
                if col != '销量':  # 避免重复处理
                    # 同样的方式处理衍生特征
                    if col in self.data.columns:
                        original_std = self.data[col].std()
                        original_mean = self.data[col].mean()
                        scenario_data[f'{col}_original'] = scenario_data[col] * original_std + original_mean
                        scenario_data[f'{col}_original'] = scenario_data[f'{col}_original'] * (1 + increase_pct / 100)
                        scenario_data[col] = (scenario_data[f'{col}_original'] - original_mean) / original_std
                        scenario_data = scenario_data.drop(f'{col}_original', axis=1)
            
            # 准备预测数据
            X_scenario = scenario_data.drop(target_col, axis=1)
            
            # 进行预测
            y_pred = self.model.predict(X_scenario)
            
            # 获取原始利润数据的均值和标准差，用于将预测结果转换回原始值
            original_profit_mean = self.data['利润'].mean()
            original_profit_std = self.data['利润'].std()
            
            # 将标准化的预测结果转换回原始利润值
            y_pred_original = y_pred * original_profit_std + original_profit_mean
            
            # 计算总利润变化
            current_total_profit = self.data['利润'].sum()
            scenario_total_profit = y_pred_original.sum()
            profit_change = scenario_total_profit - current_total_profit
            profit_change_pct = (profit_change / current_total_profit) * 100
            
            # 计算平均利润变化
            current_avg_profit = self.data['利润'].mean()
            scenario_avg_profit = y_pred_original.mean()
            avg_profit_change = scenario_avg_profit - current_avg_profit
            avg_profit_change_pct = (avg_profit_change / current_avg_profit) * 100
            
            print(f"  当前总利润: {current_total_profit:.2f}")
            print(f"  场景总利润: {scenario_total_profit:.2f}")
            print(f"  利润变化: {profit_change:.2f} ({profit_change_pct:.2f}%)")
            print(f"  当前平均利润: {current_avg_profit:.2f}")
            print(f"  场景平均利润: {scenario_avg_profit:.2f}")
            print(f"  平均利润变化: {avg_profit_change:.2f} ({avg_profit_change_pct:.2f}%)")
            
            # 保存结果
            results.append({
                '场景类型': '销售增长',
                '增长百分比': increase_pct,
                '当前总利润': current_total_profit,
                '场景总利润': scenario_total_profit,
                '总利润变化': profit_change,
                '总利润变化百分比': profit_change_pct,
                '当前平均利润': current_avg_profit,
                '场景平均利润': scenario_avg_profit,
                '平均利润变化': avg_profit_change,
                '平均利润变化百分比': avg_profit_change_pct
            })
        
        return pd.DataFrame(results)
    
    def scenario_price_change(self):
        """价格变化场景分析"""
        print("\n=== 价格变化场景分析 ===")
        
        target_col = '利润'
        
        results = []
        
        # 模拟不同的价格变化百分比
        for price_change_pct in self.config.SCENARIO_PARAMS['price_change']:
            print(f"\n价格变化 {price_change_pct}% 场景:")
            
            # 创建场景数据副本
            scenario_data = self.features_data.copy()
            
            # 检查'价格'是否是索引
            if scenario_data.index.name == '价格':
                # 将索引转换为普通列
                scenario_data = scenario_data.reset_index()
            
            # 调整价格特征
            if '价格' in scenario_data.columns:
                # 由于数据已经标准化，我们需要调整价格变化
                original_price_std = self.data['价格'].std()
                original_price_mean = self.data['价格'].mean()
                
                # 将标准化的价格转换回原始值
                scenario_data['价格_original'] = scenario_data['价格'] * original_price_std + original_price_mean
                
                # 应用价格变化百分比
                scenario_data['价格_original'] = scenario_data['价格_original'] * (1 + price_change_pct / 100)
                
                # 将调整后的价格转换回标准化值
                scenario_data['价格'] = (scenario_data['价格_original'] - original_price_mean) / original_price_std
                
                # 清理临时列
                scenario_data = scenario_data.drop('价格_original', axis=1)
            
            # 处理可能的价格衍生特征
            price_related_cols = [col for col in scenario_data.columns if '价格' in col or 'price' in col.lower()]
            for col in price_related_cols:
                if col != '价格':  # 避免重复处理
                    # 同样的方式处理衍生特征
                    if col in self.data.columns:
                        original_std = self.data[col].std()
                        original_mean = self.data[col].mean()
                        scenario_data[f'{col}_original'] = scenario_data[col] * original_std + original_mean
                        scenario_data[f'{col}_original'] = scenario_data[f'{col}_original'] * (1 + price_change_pct / 100)
                        scenario_data[col] = (scenario_data[f'{col}_original'] - original_mean) / original_std
                        scenario_data = scenario_data.drop(f'{col}_original', axis=1)
            
            # 准备预测数据
            X_scenario = scenario_data.drop(target_col, axis=1)
            
            # 进行预测
            y_pred = self.model.predict(X_scenario)
            
            # 获取原始利润数据的均值和标准差，用于将预测结果转换回原始值
            original_profit_mean = self.data['利润'].mean()
            original_profit_std = self.data['利润'].std()
            
            # 将标准化的预测结果转换回原始利润值
            y_pred_original = y_pred * original_profit_std + original_profit_mean
            
            # 计算总利润变化
            current_total_profit = self.data['利润'].sum()
            scenario_total_profit = y_pred_original.sum()
            profit_change = scenario_total_profit - current_total_profit
            profit_change_pct = (profit_change / current_total_profit) * 100
            
            # 计算平均利润变化
            current_avg_profit = self.data['利润'].mean()
            scenario_avg_profit = y_pred_original.mean()
            avg_profit_change = scenario_avg_profit - current_avg_profit
            avg_profit_change_pct = (avg_profit_change / current_avg_profit) * 100
            
            print(f"  当前总利润: {current_total_profit:.2f}")
            print(f"  场景总利润: {scenario_total_profit:.2f}")
            print(f"  利润变化: {profit_change:.2f} ({profit_change_pct:.2f}%)")
            print(f"  当前平均利润: {current_avg_profit:.2f}")
            print(f"  场景平均利润: {scenario_avg_profit:.2f}")
            print(f"  平均利润变化: {avg_profit_change:.2f} ({avg_profit_change_pct:.2f}%)")
            
            # 保存结果
            results.append({
                '场景类型': '价格变化',
                '变化百分比': price_change_pct,
                '当前总利润': current_total_profit,
                '场景总利润': scenario_total_profit,
                '总利润变化': profit_change,
                '总利润变化百分比': profit_change_pct,
                '当前平均利润': current_avg_profit,
                '场景平均利润': scenario_avg_profit,
                '平均利润变化': avg_profit_change,
                '平均利润变化百分比': avg_profit_change_pct
            })
        
        return pd.DataFrame(results)
    
    def scenario_cost_reduction(self):
        """成本降低场景分析"""
        print("\n=== 成本降低场景分析 ===")
        
        target_col = '利润'
        
        results = []
        
        # 模拟不同的成本降低百分比
        for reduction_pct in self.config.SCENARIO_PARAMS['cost_reduction']:
            print(f"\n成本降低 {reduction_pct}% 场景:")
            
            # 创建场景数据副本
            scenario_data = self.features_data.copy()
            
            # 检查'价格'是否是索引
            if scenario_data.index.name == '价格':
                # 将索引转换为普通列
                scenario_data = scenario_data.reset_index()
            
            # 调整成本特征
            if '成本' in scenario_data.columns:
                # 由于数据已经标准化，我们需要调整成本降低
                original_cost_std = self.data['成本'].std()
                original_cost_mean = self.data['成本'].mean()
                
                # 将标准化的成本转换回原始值
                scenario_data['成本_original'] = scenario_data['成本'] * original_cost_std + original_cost_mean
                
                # 应用成本降低百分比
                scenario_data['成本_original'] = scenario_data['成本_original'] * (1 - reduction_pct / 100)
                
                # 将调整后的成本转换回标准化值
                scenario_data['成本'] = (scenario_data['成本_original'] - original_cost_mean) / original_cost_std
                
                # 清理临时列
                scenario_data = scenario_data.drop('成本_original', axis=1)
            
            # 处理可能的成本衍生特征
            cost_related_cols = [col for col in scenario_data.columns if '成本' in col or 'cost' in col.lower()]
            for col in cost_related_cols:
                if col != '成本':  # 避免重复处理
                    # 同样的方式处理衍生特征
                    if col in self.data.columns:
                        original_std = self.data[col].std()
                        original_mean = self.data[col].mean()
                        scenario_data[f'{col}_original'] = scenario_data[col] * original_std + original_mean
                        scenario_data[f'{col}_original'] = scenario_data[f'{col}_original'] * (1 - reduction_pct / 100)
                        scenario_data[col] = (scenario_data[f'{col}_original'] - original_mean) / original_std
                        scenario_data = scenario_data.drop(f'{col}_original', axis=1)
            
            # 准备预测数据
            X_scenario = scenario_data.drop(target_col, axis=1)
            
            # 进行预测
            y_pred = self.model.predict(X_scenario)
            
            # 获取原始利润数据的均值和标准差，用于将预测结果转换回原始值
            original_profit_mean = self.data['利润'].mean()
            original_profit_std = self.data['利润'].std()
            
            # 将标准化的预测结果转换回原始利润值
            y_pred_original = y_pred * original_profit_std + original_profit_mean
            
            # 计算总利润变化
            current_total_profit = self.data['利润'].sum()
            scenario_total_profit = y_pred_original.sum()
            profit_change = scenario_total_profit - current_total_profit
            profit_change_pct = (profit_change / current_total_profit) * 100
            
            # 计算平均利润变化
            current_avg_profit = self.data['利润'].mean()
            scenario_avg_profit = y_pred_original.mean()
            avg_profit_change = scenario_avg_profit - current_avg_profit
            avg_profit_change_pct = (avg_profit_change / current_avg_profit) * 100
            
            print(f"  当前总利润: {current_total_profit:.2f}")
            print(f"  场景总利润: {scenario_total_profit:.2f}")
            print(f"  利润变化: {profit_change:.2f} ({profit_change_pct:.2f}%)")
            print(f"  当前平均利润: {current_avg_profit:.2f}")
            print(f"  场景平均利润: {scenario_avg_profit:.2f}")
            print(f"  平均利润变化: {avg_profit_change:.2f} ({avg_profit_change_pct:.2f}%)")
            
            # 保存结果
            results.append({
                '场景类型': '成本降低',
                '降低百分比': reduction_pct,
                '当前总利润': current_total_profit,
                '场景总利润': scenario_total_profit,
                '总利润变化': profit_change,
                '总利润变化百分比': profit_change_pct,
                '当前平均利润': current_avg_profit,
                '场景平均利润': scenario_avg_profit,
                '平均利润变化': avg_profit_change,
                '平均利润变化百分比': avg_profit_change_pct
            })
        
        return pd.DataFrame(results)
    
    def scenario_combined(self):
        """组合场景分析"""
        print("\n=== 组合场景分析 ===")
        
        target_col = '利润'
        
        # 定义组合场景
        combined_scenarios = [
            {'名称': '乐观场景', '销量增长': 20, '价格增长': 5, '成本降低': 10},
            {'名称': '保守场景', '销量增长': 5, '价格增长': 0, '成本降低': 5},
            {'名称': '稳健场景', '销量增长': 10, '价格增长': 2, '成本降低': 7}
        ]
        
        results = []
        
        for scenario in combined_scenarios:
            print(f"\n{scenario['名称']}:")
            print(f"  销量增长: {scenario['销量增长']}%, 价格增长: {scenario['价格增长']}%, 成本降低: {scenario['成本降低']}%")
            
            # 创建场景数据副本
            scenario_data = self.features_data.copy()
            
            # 调整销量特征
            if '销量' in scenario_data.columns:
                # 由于数据已经标准化，我们需要调整增加量
                original_sales_std = self.data['销量'].std()
                original_sales_mean = self.data['销量'].mean()
                
                # 将标准化的销量转换回原始值
                scenario_data['销量_original'] = scenario_data['销量'] * original_sales_std + original_sales_mean
                
                # 应用增长百分比
                scenario_data['销量_original'] = scenario_data['销量_original'] * (1 + scenario['销量增长'] / 100)
                
                # 将调整后的销量转换回标准化值
                scenario_data['销量'] = (scenario_data['销量_original'] - original_sales_mean) / original_sales_std
                
                # 清理临时列
                scenario_data = scenario_data.drop('销量_original', axis=1)
            
            # 调整价格特征
            if '价格' in scenario_data.columns:
                # 由于数据已经标准化，我们需要调整价格变化
                original_price_std = self.data['价格'].std()
                original_price_mean = self.data['价格'].mean()
                
                # 将标准化的价格转换回原始值
                scenario_data['价格_original'] = scenario_data['价格'] * original_price_std + original_price_mean
                
                # 应用价格变化百分比
                scenario_data['价格_original'] = scenario_data['价格_original'] * (1 + scenario['价格增长'] / 100)
                
                # 将调整后的价格转换回标准化值
                scenario_data['价格'] = (scenario_data['价格_original'] - original_price_mean) / original_price_std
                
                # 清理临时列
                scenario_data = scenario_data.drop('价格_original', axis=1)
            
            # 调整成本特征
            if '成本' in scenario_data.columns:
                # 由于数据已经标准化，我们需要调整成本降低
                original_cost_std = self.data['成本'].std()
                original_cost_mean = self.data['成本'].mean()
                
                # 将标准化的成本转换回原始值
                scenario_data['成本_original'] = scenario_data['成本'] * original_cost_std + original_cost_mean
                
                # 应用成本降低百分比
                scenario_data['成本_original'] = scenario_data['成本_original'] * (1 - scenario['成本降低'] / 100)
                
                # 将调整后的成本转换回标准化值
                scenario_data['成本'] = (scenario_data['成本_original'] - original_cost_mean) / original_cost_std
                
                # 清理临时列
                scenario_data = scenario_data.drop('成本_original', axis=1)
            
            # 准备预测数据
            X_scenario = scenario_data.drop(target_col, axis=1)
            
            # 进行预测
            y_pred = self.model.predict(X_scenario)
            
            # 获取原始利润数据的均值和标准差，用于将预测结果转换回原始值
            original_profit_mean = self.data['利润'].mean()
            original_profit_std = self.data['利润'].std()
            
            # 将标准化的预测结果转换回原始利润值
            y_pred_original = y_pred * original_profit_std + original_profit_mean
            
            # 计算总利润变化
            current_total_profit = self.data['利润'].sum()
            scenario_total_profit = y_pred_original.sum()
            profit_change = scenario_total_profit - current_total_profit
            profit_change_pct = (profit_change / current_total_profit) * 100
            
            print(f"  当前总利润: {current_total_profit:.2f}")
            print(f"  场景总利润: {scenario_total_profit:.2f}")
            print(f"  利润变化: {profit_change:.2f} ({profit_change_pct:.2f}%)")
            
            # 保存结果
            results.append({
                '场景名称': scenario['名称'],
                '销量增长百分比': scenario['销量增长'],
                '价格增长百分比': scenario['价格增长'],
                '成本降低百分比': scenario['成本降低'],
                '当前总利润': current_total_profit,
                '场景总利润': scenario_total_profit,
                '总利润变化': profit_change,
                '总利润变化百分比': profit_change_pct
            })
        
        return pd.DataFrame(results)
    
    def generate_scenario_report(self):
        """生成场景分析报告"""
        print("\n=== 生成场景分析报告 ===")
        
        # 运行所有场景分析
        sales_results = self.scenario_sales_increase()
        price_results = self.scenario_price_change()
        cost_results = self.scenario_cost_reduction()
        combined_results = self.scenario_combined()
        
        # 保存所有结果
        sales_file = os.path.join(self.config.RESULTS_DIR, 'scenario_sales_increase.csv')
        price_file = os.path.join(self.config.RESULTS_DIR, 'scenario_price_change.csv')
        cost_file = os.path.join(self.config.RESULTS_DIR, 'scenario_cost_reduction.csv')
        combined_file = os.path.join(self.config.RESULTS_DIR, 'scenario_combined.csv')
        
        sales_results.to_csv(sales_file, index=False)
        price_results.to_csv(price_file, index=False)
        cost_results.to_csv(cost_file, index=False)
        combined_results.to_csv(combined_file, index=False)
        
        print(f"\n场景分析结果已保存到:")
        print(f"- 销售增长场景: {sales_file}")
        print(f"- 价格变化场景: {price_file}")
        print(f"- 成本降低场景: {cost_file}")
        print(f"- 组合场景: {combined_file}")
        
        # 生成综合报告
        report_file = os.path.join(self.config.RESULTS_DIR, 'scenario_analysis_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 预案性分析报告\n\n")
            f.write("## 1. 分析概述\n")
            f.write("本报告对不同业务场景下的利润变化进行了预测分析，包括销售增长、价格变化、成本降低以及组合场景。\n\n")
            
            f.write("## 2. 当前基线\n")
            baseline = self.get_current_baseline()
            for key, value in baseline.items():
                if '平均' in key or '总' in key:
                    f.write(f"- {key}: {value:.2f}\n")
                else:
                    f.write(f"- {key}: {value}\n")
            f.write("\n")
            
            f.write("## 3. 销售增长场景\n")
            f.write("### 3.1 场景假设\n")
            f.write("在保持价格和成本不变的情况下，分析不同销售增长百分比对利润的影响。\n\n")
            
            f.write("### 3.2 分析结果\n")
            for _, row in sales_results.iterrows():
                f.write(f"**销售增长 {row['增长百分比']}%**：\n")
                f.write(f"- 当前总利润: {row['当前总利润']:.2f}\n")
                f.write(f"- 场景总利润: {row['场景总利润']:.2f}\n")
                f.write(f"- 利润变化: {row['总利润变化']:.2f} ({row['总利润变化百分比']:.2f}%)\n\n")
            
            f.write("## 4. 价格变化场景\n")
            f.write("### 4.1 场景假设\n")
            f.write("在保持销量和成本不变的情况下，分析不同价格变化百分比对利润的影响。\n\n")
            
            f.write("### 4.2 分析结果\n")
            for _, row in price_results.iterrows():
                f.write(f"**价格变化 {row['变化百分比']}%**：\n")
                f.write(f"- 当前总利润: {row['当前总利润']:.2f}\n")
                f.write(f"- 场景总利润: {row['场景总利润']:.2f}\n")
                f.write(f"- 利润变化: {row['总利润变化']:.2f} ({row['总利润变化百分比']:.2f}%)\n\n")
            
            f.write("## 5. 成本降低场景\n")
            f.write("### 5.1 场景假设\n")
            f.write("在保持销量和价格不变的情况下，分析不同成本降低百分比对利润的影响。\n\n")
            
            f.write("### 5.2 分析结果\n")
            for _, row in cost_results.iterrows():
                f.write(f"**成本降低 {row['降低百分比']}%**：\n")
                f.write(f"- 当前总利润: {row['当前总利润']:.2f}\n")
                f.write(f"- 场景总利润: {row['场景总利润']:.2f}\n")
                f.write(f"- 利润变化: {row['总利润变化']:.2f} ({row['总利润变化百分比']:.2f}%)\n\n")
            
            f.write("## 6. 组合场景\n")
            f.write("### 6.1 场景假设\n")
            f.write("同时考虑销量、价格和成本的变化，分析综合因素对利润的影响。\n\n")
            
            f.write("### 6.2 分析结果\n")
            for _, row in combined_results.iterrows():
                f.write(f"**{row['场景名称']}**：\n")
                f.write(f"- 销量增长: {row['销量增长百分比']}%, 价格增长: {row['价格增长百分比']}%, 成本降低: {row['成本降低百分比']}%\n")
                f.write(f"- 当前总利润: {row['当前总利润']:.2f}\n")
                f.write(f"- 场景总利润: {row['场景总利润']:.2f}\n")
                f.write(f"- 利润变化: {row['总利润变化']:.2f} ({row['总利润变化百分比']:.2f}%)\n\n")
            
            f.write("## 7. 结论与建议\n")
            f.write("1. **成本降低对利润的影响最直接**：降低成本可以直接提高利润，建议优先考虑成本优化措施。\n")
            f.write("2. **销售增长是长期利润增长的关键**：通过市场扩展、营销活动等方式提高销量，可以带来持续的利润增长。\n")
            f.write("3. **价格调整需谨慎**：价格上涨可能会影响销量，需要综合考虑市场需求弹性。\n")
            f.write("4. **组合策略效果最佳**：同时实施多项优化措施，可以获得最大的利润增长。\n")
        
        print(f"\n场景分析报告已生成: {report_file}")
        
        return {
            'sales_results': sales_results,
            'price_results': price_results,
            'cost_results': cost_results,
            'combined_results': combined_results
        }
    
    def run_scenario_analysis(self):
        """运行完整的预案性分析流程"""
        print("\n=== 开始预案性分析流程 ===")
        
        # 1. 加载数据和模型
        self.load_data_and_model()
        
        # 2. 获取当前基线
        self.get_current_baseline()
        
        # 3. 生成场景分析报告
        results = self.generate_scenario_report()
        
        print("\n=== 预案性分析流程完成 ===")
        return results

if __name__ == "__main__":
    scenario_analyzer = ScenarioAnalyzer()
    scenario_analyzer.run_scenario_analysis()