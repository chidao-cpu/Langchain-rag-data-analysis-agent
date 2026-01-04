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
        self.data_info = None
        self.scenario_suggestions = None
    
    def load_data_and_model(self):
        """加载数据和训练好的模型"""
        # 加载预处理后的数据
        if os.path.exists(self.config.PROCESSED_DATA_FILE):
            self.data = pd.read_csv(self.config.PROCESSED_DATA_FILE)
            # 动态检测日期时间列
            for col in self.data.columns:
                if pd.api.types.is_datetime64_any_dtype(self.data[col]) or '日期' in col or 'time' in col.lower():
                    try:
                        self.data[col] = pd.to_datetime(self.data[col])
                    except:
                        pass
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
    
    def collect_data_info(self):
        """收集数据信息作为API请求的辅助prompt"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        data_info = {
            'shape': f"数据形状: {self.data.shape[0]} 行, {self.data.shape[1]} 列",
            'columns': f"列名: {', '.join(self.data.columns.tolist())}",
            'data_types': "数据类型:\n" + '\n'.join([f"  {col}: {self.data[col].dtype}" for col in self.data.columns]),
            'missing_values': "缺失值情况:\n" + '\n'.join([f"  {col}: {self.data[col].isnull().sum()} ({self.data[col].isnull().sum()/len(self.data)*100:.1f}%)" for col in self.data.columns if self.data[col].isnull().sum() > 0]),
            'numerical_columns': f"数值型列: {', '.join(self.data.select_dtypes(include=[np.number]).columns.tolist())}",
            'categorical_columns': f"类别型列: {', '.join(self.data.select_dtypes(include=['object', 'category']).columns.tolist())}",
            'datetime_columns': f"日期时间列: {', '.join(self.data.select_dtypes(include=['datetime64']).columns.tolist())}"
        }
        
        # 添加数值型列的统计信息
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            stats = self.data[numerical_cols].describe().round(2)
            data_info['numerical_stats'] = "数值型列统计信息:\n" + stats.to_string()
        
        # 添加类别型列的唯一值信息
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            unique_values = "类别型列唯一值:\n"
            for col in categorical_cols:
                unique_vals = self.data[col].unique()
                # 限制显示的唯一值数量
                if len(unique_vals) > 10:
                    unique_vals = unique_vals[:10] + ['...']
                unique_values += f"  {col}: {', '.join(map(str, unique_vals))}\n"
            data_info['categorical_unique'] = unique_values
        
        self.data_info = data_info
        return data_info
    
    def get_scenario_suggestions(self):
        """调用DeepSeek API获取场景分析建议"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 如果数据信息未收集，则先收集
        if self.data_info is None:
            self.data_info = self.collect_data_info()
        
        print(f"\n开始获取场景分析建议...")
        
        # 初始化DeepSeek API客户端（动态导入以避免循环导入）
        from src.langchain_agent import OpenAIClientLLM
        deepseek_client = OpenAIClientLLM()
        
        # 构建data_info字符串
        data_info_str = "\n".join([f"{key}: {value}" for key, value in self.data_info.items()])
        
        # 读取docs/api_docs中的文件内容
        try:
            with open("docs/api_docs/data_preprocessing.md", "r", encoding="utf-8") as f:
                data_preprocessing_content = f.read()
            
            with open("docs/api_docs/feature_engineering.md", "r", encoding="utf-8") as f:
                feature_engineering_content = f.read()
            
            with open("docs/api_docs/regression_analysis.md", "r", encoding="utf-8") as f:
                regression_analysis_content = f.read()
        except Exception as e:
            print(f"  读取API文档失败：{str(e)}")
            data_preprocessing_content = ""
            feature_engineering_content = ""
            regression_analysis_content = ""
        
        # 构建prompt
        prompt = f"你是一个数据分析专家，请根据以下数据集信息和完整的数据分析结果，建议合适的场景分析方式：\n\n"
        prompt += data_info_str
        prompt += f"\n=== 已完成的数据分析环节详细结果 ===\n"
        prompt += f"1. 数据预处理结果：\n{data_preprocessing_content}\n\n"
        prompt += f"2. 特征工程和可视化分析结果：\n{feature_engineering_content}\n\n"
        prompt += f"3. 回归分析结果：\n{regression_analysis_content}\n\n"
        prompt += f"请基于以上完整的数据分析结果，为我生成适合该数据集的通用场景分析建议，要求：\n"
        prompt += f"1. 场景分析建议应覆盖不同的业务维度，如市场策略、销售增长、价格变化、资源分配、风险评估、成本优化、收益增长等(选择四种左右即可)\n\n"
        prompt += f"2. 避免仅针对单一数据集的特定业务场景"   
        prompt += f"3. 场景类型使用通用术语，如market_expansion, resource_allocation, risk_assessment等\n"
        prompt += f"4. 每个场景应包含可量化的参数，如变化百分比、阈值、范围等\n"
        prompt += f"5. 场景建议应与前面四个阶段的数据分析结果紧密结合\n\n"
        prompt += f"请以JSON格式输出建议的场景分析方式，每个场景分析应包含：\n"
        prompt += f"- scenario_type: 场景类型（通用术语）\n"
        prompt += f"- title: 场景标题\n"
        prompt += f"- parameters: 场景参数（如变化百分比、阈值、范围等）\n"
        prompt += f"- description: 场景描述\n"
        prompt += f"- recommended: 是否推荐（true/false）\n"
        prompt += f"\n例如：\n"
        prompt += f"[{{\"scenario_type\": \"market_expansion\", \"title\": \"市场扩张场景\", \"parameters\": {{\"growth_rate\": [10, 20, 30]}}, \"description\": \"模拟不同市场增长率对关键指标的影响\", \"recommended\": true}}]\n"
        prompt += f"\n请只输出JSON格式的结果，不要添加任何其他文字或解释。\n"
        
        try:
            # 调用DeepSeek API获取建议
            print("  正在调用DeepSeek API获取场景分析建议...")
            response = deepseek_client.invoke(prompt)
            
            # 解析API响应
            import json
            import re
            
            # 清理响应内容，去除可能的Markdown格式
            if isinstance(response, str):
                # 去除可能的Markdown代码块标记
                response = re.sub(r'^```json\n', '', response)
                response = re.sub(r'\n```$', '', response)
                # 去除多余的空格和换行符
                response = response.strip()
                
            self.scenario_suggestions = json.loads(response)
            
           
                
            print(f"  成功获取{len(self.scenario_suggestions)}个场景分析建议")
            return self.scenario_suggestions
            
        except Exception as e:
            print(f"  调用DeepSeek API失败：{str(e)}")
            import traceback
            traceback.print_exc()
            # API调用失败时返回空列表，不再使用默认建议
            self.scenario_suggestions = []
            return self.scenario_suggestions
    
    def get_current_baseline(self):
        """获取当前基线数据"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 检测可能的列
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 检测具体列
        price_cols = [col for col in numerical_cols if '价格' in col or 'price' in col.lower()]
        sales_cols = [col for col in numerical_cols if '销量' in col or 'sales' in col.lower() or 'quantity' in col.lower()]
        cost_cols = [col for col in numerical_cols if '成本' in col or 'cost' in col.lower()]
        profit_cols = [col for col in numerical_cols if '利润' in col or 'profit' in col.lower() or 'income' in col.lower()]
        revenue_cols = [col for col in numerical_cols if '销售额' in col or 'revenue' in col.lower()]
        product_cols = [col for col in categorical_cols if '产品' in col or 'product' in col.lower()]
        region_cols = [col for col in categorical_cols if '地区' in col or '区域' in col or 'region' in col.lower()]
        
        baseline = {}
        
        # 添加平均价格
        if price_cols:
            baseline[f'平均{price_cols[0]}'] = self.data[price_cols[0]].mean()
        
        # 添加平均销量
        if sales_cols:
            baseline[f'平均{sales_cols[0]}'] = self.data[sales_cols[0]].mean()
        
        # 添加平均成本
        if cost_cols:
            baseline[f'平均{cost_cols[0]}'] = self.data[cost_cols[0]].mean()
        
        # 添加平均利润
        if profit_cols:
            baseline[f'平均{profit_cols[0]}'] = self.data[profit_cols[0]].mean()
        
        # 添加总销售额
        if revenue_cols:
            baseline[f'总{revenue_cols[0]}'] = self.data[revenue_cols[0]].sum()
        
        # 添加总利润
        if profit_cols:
            baseline[f'总{profit_cols[0]}'] = self.data[profit_cols[0]].sum()
        
        # 添加产品类别数
        if product_cols:
            baseline[f'{product_cols[0]}数'] = self.data[product_cols[0]].nunique()
        
        # 添加地区数
        if region_cols:
            baseline[f'{region_cols[0]}数'] = self.data[region_cols[0]].nunique()
        
        print("\n=== 当前基线数据 ===")
        for key, value in baseline.items():
            if '平均' in key or '总' in key:
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        return baseline

  
    def generate_scenario_report(self):
        """生成场景分析报告"""
        print("\n=== 生成场景分析报告 ===")
        
        # 检查是否有场景建议
        if self.scenario_suggestions is None or len(self.scenario_suggestions) == 0:
            print("没有获取到场景分析建议，无法生成报告。")
            return {}
        
        # 获取当前基线
        baseline = self.get_current_baseline()
        
        # 获取数值型列，用于场景分析
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 为每个场景生成分析结果
        all_results = {}
        
        print("\n开始生成各场景分析结果...")
        
        # 创建结果目录（如果不存在）
        if not os.path.exists(self.config.RESULTS_DIR):
            os.makedirs(self.config.RESULTS_DIR)
        
        # 生成每个场景的分析结果
        for i, scenario in enumerate(self.scenario_suggestions):
            if not scenario.get('recommended', True):
                continue
            
            scenario_type = scenario.get('scenario_type', f'scenario_{i+1}')
            title = scenario.get('title', f'场景{i+1}')
            parameters = scenario.get('parameters', {})
            description = scenario.get('description', '')
            
            print(f"  生成{title}分析结果...")
            
            # 创建一个简单的场景结果DataFrame
            # 注意：这是一个简化的实现，实际项目中可能需要根据具体场景和参数进行更复杂的分析
            scenario_results = pd.DataFrame()
            
            # 为每个参数组合生成结果
            # 这里假设parameters是一个字典，值是列表形式的参数范围
            for param_name, param_values in parameters.items():
                if not isinstance(param_values, list):
                    param_values = [param_values]
                
                for value in param_values:
                    # 创建一个新行表示这个参数组合的场景
                    scenario_data = self.data.copy()
                    
                    # 根据参数名和值调整数据
                    if isinstance(value, (int, float)):
                        # 参数名到列名的映射（包含API实际返回的参数）
                        param_to_cols = {
                            'profit_margin_volatility_threshold': ['利润率', 'profit', 'margin'],
                            'allocation_adjustment_range': ['销售额', '销量', 'revenue', 'sales'],
                            'price_change_range': ['价格', 'price'],
                            'cost_reduction_targets': ['成本', 'cost'],
                            'price_cost_ratio_threshold': ['价格成本比', 'price_cost_ratio'],
                            'promotion_profit_margin_threshold': ['利润率', 'profit_margin'],
                            'sales_volume_increase_targets': ['销量', 'sales', 'volume'],
                            'weekend_vs_weekday_multiplier': ['销量', '销售额', 'sales', 'revenue'],
                            'monthly_sales_variation_range': ['销量', '销售额', 'sales', 'revenue'],
                            'profit_margin_volatility_threshold': ['利润率', 'profit_margin'],
                            'worst_case_scenario_drop': ['销量', '销售额', 'sales', 'revenue'],
                            'product_mix_profitability_target': ['利润率', 'profit_margin'],
                            'cross_sell_ratio_increase': ['销量', '销售额', 'sales', 'revenue'],
                            'low_performer_cutoff_threshold': ['利润率', 'profit_margin'],
                            'cost_to_sales_ratio_targets': ['成本', 'cost', '销售额', 'sales'],
                            'efficiency_improvement_potential': ['成本', 'cost'],
                            'region_performance_gap_threshold': ['销售额', '利润', 'sales', 'profit'],
                            'key_performance_indicators': ['销售额', '利润', '利润率', 'sales', 'profit', 'margin']
                        }
                        
                        # 确定要调整的列
                        target_cols = []
                        
                        # 1. 先检查是否有参数名的映射
                        if param_name in param_to_cols:
                            for col in numerical_cols:
                                for keyword in param_to_cols[param_name]:
                                    if keyword in col or keyword in col.lower():
                                        target_cols.append(col)
                                        break
                        else:
                            # 2. 直接匹配参数名中的关键词
                            param_keywords = param_name.replace('_', ' ').split()
                            for col in numerical_cols:
                                for keyword in param_keywords:
                                    if keyword in col or keyword in col.lower():
                                        target_cols.append(col)
                                        break
                            
                            # 3. 如果没有找到，检查是否是成本相关
                            if not target_cols and ('cost' in param_name.lower() or 'reduce' in param_name.lower()):
                                for col in numerical_cols:
                                    if '成本' in col or 'cost' in col.lower():
                                        target_cols.append(col)
                        
                        # 4. 如果还是没有找到，使用默认的销售额和利润列
                        if not target_cols:
                            target_cols = [col for col in numerical_cols if '销售额' in col or '利润' in col or 'revenue' in col or 'profit' in col]
                        
                        # 5. 执行调整
                        if target_cols:
                            for col in target_cols:
                                # 根据参数类型应用不同的调整逻辑
                                if 'range' in param_name.lower():
                                    # 范围参数 - 应用百分比变化
                                    scenario_data[col] *= (1 + value / 100)
                                elif 'threshold' in param_name.lower():
                                    # 阈值参数 - 对低于阈值的行进行调整
                                    scenario_data.loc[scenario_data[col] < value, col] *= 1.05
                                elif 'target' in param_name.lower() or 'targets' in param_name.lower():
                                    # 目标参数 - 应用百分比变化
                                    scenario_data[col] *= (1 + value / 100)
                                elif 'margin' in param_name.lower() or 'profit' in param_name.lower():
                                    # 利润率相关参数 - 直接调整利润率
                                    scenario_data[col] += value / 100
                                elif 'cost' in param_name.lower() or 'reduction' in param_name.lower():
                                    # 成本相关参数 - 减少成本
                                    scenario_data[col] *= (1 - value / 100)
                                elif 'increase' in param_name.lower() or 'growth' in param_name.lower():
                                    # 增长相关参数 - 增加数值
                                    scenario_data[col] *= (1 + value / 100)
                                elif 'multiplier' in param_name.lower():
                                    # 乘数参数 - 直接乘以乘数
                                    scenario_data[col] *= value
                                else:
                                    # 默认 - 应用百分比变化
                                    scenario_data[col] *= (1 + value / 100)
                    
                    # 计算当前基线和场景结果
                    # 优先使用总销售额，如果没有则使用总利润
                    current_total = baseline.get('总销售额', baseline.get('总利润', 0))
                    # 使用销售额或利润列的总和作为场景总值
                    revenue_profit_cols = [col for col in numerical_cols if '销售额' in col or '利润' in col]
                    if revenue_profit_cols:
                        scenario_total = scenario_data[revenue_profit_cols].sum().sum()
                    else:
                        scenario_total = scenario_data[numerical_cols].sum().sum() if numerical_cols else 0
                    change = scenario_total - current_total
                    change_pct = (change / current_total * 100) if current_total != 0 else 0
                    
                    # 添加到结果DataFrame
                    scenario_results = pd.concat([
                        scenario_results,
                        pd.DataFrame({
                            '参数名称': [param_name],
                            '参数值': [value],
                            '当前总值': [current_total],
                            '场景总值': [scenario_total],
                            '变化值': [change],
                            '变化百分比': [change_pct]
                        })
                    ], ignore_index=True)
            
            # 保存结果到字典（不再保存为单独的CSV文件）
            all_results[scenario_type] = scenario_results
        
        # 生成综合报告
        report_file = os.path.join('docs/api_docs', 'scenario_analysis_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 预案性分析报告\n\n")
            f.write("## 1. 分析概述\n")
            f.write(f"本报告对{len(all_results)}个动态生成的业务场景进行了预测分析，基于DeepSeek API提供的场景建议。\n\n")
            
            f.write("## 2. 当前基线\n")
            for key, value in baseline.items():
                if '平均' in key or '总' in key:
                    f.write(f"- {key}: {value:.2f}\n")
                else:
                    f.write(f"- {key}: {value}\n")
            f.write("\n")
            
            # 为每个场景生成报告部分
            for i, scenario in enumerate(self.scenario_suggestions):
                if not scenario.get('recommended', True):
                    continue
                
                scenario_type = scenario.get('scenario_type', f'scenario_{i+1}')
                title = scenario.get('title', f'场景{i+1}')
                parameters = scenario.get('parameters', {})
                description = scenario.get('description', '')
                
                f.write(f"## {i+3}. {title}\n")
                f.write("### {i+3}.1 场景假设\n")
                f.write(f"{description}\n\n")
                
                # 添加参数信息
                if parameters:
                    f.write("#### 场景参数\n")
                    for param_name, param_values in parameters.items():
                        f.write(f"- {param_name}: {param_values}\n")
                    f.write("\n")
                
                f.write("### {i+3}.2 分析结果\n")
                
                # 获取该场景的结果
                if scenario_type in all_results:
                    scenario_results = all_results[scenario_type]
                    
                    if not scenario_results.empty:
                        for _, row in scenario_results.iterrows():
                            f.write(f"**{row['参数名称']}: {row['参数值']}**：\n")
                            f.write(f"- 当前总值: {row['当前总值']:.2f}\n")
                            f.write(f"- 场景总值: {row['场景总值']:.2f}\n")
                            f.write(f"- 变化值: {row['变化值']:.2f}\n")
                            f.write(f"- 变化百分比: {row['变化百分比']:.2f}%\n\n")
                    else:
                        f.write("未生成有效分析结果\n\n")
                else:
                    f.write("未找到场景分析结果\n\n")
            
            f.write("## " + str(len(self.scenario_suggestions) + 3) + ". 结论与建议\n")
            f.write("1. **场景分析建议**：基于DeepSeek API生成的场景分析建议，覆盖了不同的业务维度和参数组合。\n")
            f.write("2. **参数敏感性**：不同参数对结果的影响程度不同，建议重点关注影响最大的参数。\n")
            f.write("3. **策略选择**：根据业务目标和资源情况，选择最适合的场景策略实施。\n")
            f.write("4. **持续优化**：建议定期更新场景分析，以适应市场变化和业务发展。\n")
        
        print(f"\n场景分析报告已生成: {report_file}")
        
        return all_results
    
    def run_scenario_analysis(self):
        """运行完整的预案性分析流程"""
        print("\n=== 开始预案性分析流程 ===")
        
        # 1. 加载数据和模型
        self.load_data_and_model()
        
        # 2. 获取当前基线
        self.get_current_baseline()
        
        # 3. 获取场景分析建议
        self.get_scenario_suggestions()
        
        # 4. 生成场景分析报告
        results = self.generate_scenario_report()
        
        print("\n=== 预案性分析流程完成 ===")
        return results

if __name__ == "__main__":
    scenario_analyzer = ScenarioAnalyzer()
    scenario_analyzer.run_scenario_analysis()