import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import Config
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class DataVisualizer:
    def __init__(self):
        self.config = Config()
        self.data = None
        self.features_data = None
        self.data_info = None
        self.visualization_suggestions = None
        
    def load_data(self, processed_file=None, features_file=None):
        """加载数据"""
        # 加载预处理后的数据
        processed_file = processed_file or self.config.PROCESSED_DATA_FILE
        if os.path.exists(processed_file):
            self.data = pd.read_csv(processed_file)
            # 检测并转换日期时间列
            for col in self.data.columns:
                if pd.api.types.is_datetime64_any_dtype(self.data[col]) or '日期' in col or 'time' in col.lower():
                    try:
                        self.data[col] = pd.to_datetime(self.data[col])
                    except:
                        pass
            print(f"成功加载预处理后的数据: {processed_file}")
        
        # 加载特征工程后的数据
        features_file = features_file or self.config.FEATURES_DATA_FILE
        if os.path.exists(features_file):
            self.features_data = pd.read_csv(features_file)
            print(f"成功加载特征工程后的数据: {features_file}")
        
        return self.data, self.features_data
    
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
    
    def get_visualization_suggestions(self):
        """调用DeepSeek API获取可视化建议"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 如果数据信息未收集，则先收集
        if self.data_info is None:
            self.data_info = self.collect_data_info()
        
        print(f"\n开始获取可视化建议...")
        
        # 初始化DeepSeek API客户端（动态导入以避免循环导入）
        from src.langchain_agent import OpenAIClientLLM
        deepseek_client = OpenAIClientLLM()
        
        # 构建data_info字符串
        data_info_str = "\n".join([f"{key}: {value}" for key, value in self.data_info.items()])
        
        # 构建prompt
        prompt = f"你是一个数据可视化专家，请根据以下数据集信息，建议合适的可视化方式：\n\n"
        prompt += data_info_str
        prompt += f"\n请以JSON格式输出建议的可视化方式，每个可视化应包含：\n"
        prompt += f"1. chart_type: 图表类型（如histogram, scatter, boxplot等）\n"
        prompt += f"2. title: 图表标题\n"
        prompt += f"3. x_axis: X轴字段\n"
        prompt += f"4. y_axis: Y轴字段\n"
        prompt += f"5. description: 图表描述\n"
        prompt += f"6. recommended: 是否推荐（true/false）\n"
        prompt += f"\n例如：\n"
        prompt += f"[{{\"chart_type\": \"histogram\", \"title\": \"价格分布\", \"x_axis\": \"价格\", \"y_axis\": \"频率\", \"description\": \"产品价格的分布情况\", \"recommended\": true}}]\n"
        prompt += f"\n请只输出JSON格式的结果，不要添加任何其他文字或解释。\n"
        
        try:
            # 调用DeepSeek API获取建议
            print("  正在调用DeepSeek API获取可视化建议...")
            response = deepseek_client.invoke(prompt)
            
            # 解析API响应
            import json
            self.visualization_suggestions = json.loads(response)
            
            print(f"  成功获取{len(self.visualization_suggestions)}个可视化建议")
            return self.visualization_suggestions
            
        except Exception as e:
            print(f"  调用DeepSeek API失败：{str(e)}")
            self.visualization_suggestions = []
            return self.visualization_suggestions
    
    

    
    def plot_from_suggestions(self):
        """根据可视化建议生成图表"""
        if self.visualization_suggestions is None:
            print("未获取到可视化建议，使用默认可视化流程")
            return
        
        print(f"\n根据可视化建议生成图表...")
        
        # 为不同图表类型分配处理函数
        chart_handlers = {
            'histogram': self._plot_histogram,
            'scatter': self._plot_scatter,
            'boxplot': self._plot_boxplot,
            'bar': self._plot_bar,
            'line': self._plot_line,
            'heatmap': self.plot_correlation_heatmap,
            'pairplot': self.plot_scatter_matrix,
            'feature_importance': self.plot_feature_importance
        }
        
        # 跟踪已生成的图表，避免重复
        self.generated_charts = set()
        
        for suggestion in self.visualization_suggestions:
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
                if x_axis and x_axis not in self.data.columns:
                    print(f"跳过{title}: X轴字段{x_axis}不存在")
                    continue
                if y_axis and y_axis not in self.data.columns:
                    print(f"跳过{title}: Y轴字段{y_axis}不存在")
                    continue
                
                # 生成图表唯一标识
                chart_id = f"{chart_type}_{x_axis or 'none'}_{y_axis or 'none'}"
                if chart_id in self.generated_charts:
                    continue
                self.generated_charts.add(chart_id)
                
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
        
    def _plot_histogram(self, x_axis, y_axis, title, description):
        """绘制直方图"""
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data[x_axis], bins=20, kde=True, color='skyblue')
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.config.VISUALIZATIONS_DIR, f'{title}_histogram.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    直方图已保存到: {output_path}")
    
    def _plot_scatter(self, x_axis, y_axis, title, description):
        """绘制散点图"""
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=self.data, x=x_axis, y=y_axis, color='orange', alpha=0.7)
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.config.VISUALIZATIONS_DIR, f'{title}_scatter.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    散点图已保存到: {output_path}")
    
    def _plot_boxplot(self, x_axis, y_axis, title, description):
        """绘制箱线图"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.data, x=x_axis, y=y_axis, palette='Set2')
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.config.VISUALIZATIONS_DIR, f'{title}_boxplot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    箱线图已保存到: {output_path}")
    
    def _plot_bar(self, x_axis, y_axis, title, description):
        """绘制柱状图"""
        plt.figure(figsize=(12, 6))
        sns.barplot(data=self.data, x=x_axis, y=y_axis, palette='Set3')
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.config.VISUALIZATIONS_DIR, f'{title}_bar.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    柱状图已保存到: {output_path}")
    
    def _plot_line(self, x_axis, y_axis, title, description):
        """绘制折线图"""
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.data, x=x_axis, y=y_axis, color='green', marker='o')
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.config.VISUALIZATIONS_DIR, f'{title}_line.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    折线图已保存到: {output_path}")

    def write_visualization_report(self):
        """将可视化结果写入到feature_engineering.md文档"""
        if self.visualization_suggestions is None:
            raise ValueError("请先获取可视化建议")
        
        # 构建报告内容
        report_content = "\n## 数据可视化结果\n\n"
        report_content += "### 可视化建议\n\n"
        
        if self.visualization_suggestions:
            for i, suggestion in enumerate(self.visualization_suggestions):
                if not suggestion.get('recommended', True):
                    continue
                
                chart_type = suggestion.get('chart_type', '未知')
                title = suggestion.get('title', '无标题')
                x_axis = suggestion.get('x_axis', '无')
                y_axis = suggestion.get('y_axis', '无')
                description = suggestion.get('description', '无描述')
                
                report_content += f"{i+1}. **{chart_type}** - {title}\n"
                report_content += f"   - X轴: {x_axis}\n"
                report_content += f"   - Y轴: {y_axis}\n"
                report_content += f"   - 描述: {description}\n\n"
        else:
            report_content += "未获取到可视化建议\n\n"
        
        # 写入到feature_engineering.md文件
        report_path = "docs/api_docs/data_visualization.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n可视化结果已写入到: {report_path}")
        return report_path
    
    def run_all_visualizations(self):
        """运行所有可视化"""
        print("=== 开始数据可视化流程 ===")
        
        # 加载数据
        self.load_data()
        
        # 收集数据信息并获取可视化建议
        self.collect_data_info()
        self.get_visualization_suggestions()
        
        # 根据可视化建议生成图表
        self.plot_from_suggestions()
        
        # 将可视化结果写入文档
        self.write_visualization_report()
        
        print("\n=== 数据可视化流程完成 ===")

if __name__ == "__main__":
    visualizer = DataVisualizer()
    visualizer.run_all_visualizations()
