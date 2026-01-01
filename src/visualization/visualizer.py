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
        
    def load_data(self, processed_file=None, features_file=None):
        """加载数据"""
        # 加载预处理后的数据
        processed_file = processed_file or self.config.PROCESSED_DATA_FILE
        if os.path.exists(processed_file):
            self.data = pd.read_csv(processed_file)
            if '销售日期' in self.data.columns:
                self.data['销售日期'] = pd.to_datetime(self.data['销售日期'])
            print(f"成功加载预处理后的数据: {processed_file}")
        
        # 加载特征工程后的数据
        features_file = features_file or self.config.FEATURES_DATA_FILE
        if os.path.exists(features_file):
            self.features_data = pd.read_csv(features_file)
            print(f"成功加载特征工程后的数据: {features_file}")
        
        return self.data, self.features_data
    
    def plot_numerical_distribution(self, numerical_cols=None):
        """绘制数值型特征的分布直方图"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        numerical_cols = numerical_cols or ['价格', '销量', '成本', '客户评分', '销售额', '利润', '利润率']
        numerical_cols = [col for col in numerical_cols if col in self.data.columns]
        
        n_cols = 2
        n_rows = (len(numerical_cols) + 1) // 2
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, col in enumerate(numerical_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(self.data[col], bins=20, kde=True, color='skyblue')
            plt.title(f'{col}的分布')
            plt.xlabel(col)
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.config.VISUALIZATIONS_DIR, 'numerical_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"数值型特征分布直方图已保存到: {output_path}")
    
    def plot_correlation_heatmap(self):
        """绘制相关性热力图"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 选择数值型特征
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        plt.figure(figsize=(12, 10))
        corr_matrix = self.data[numerical_cols].corr()
        
        # 使用mask只显示下三角
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=.5, cbar_kws={'shrink': .8})
        plt.title('特征相关性热力图', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        output_path = os.path.join(self.config.VISUALIZATIONS_DIR, 'correlation_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"相关性热力图已保存到: {output_path}")
    
    def plot_time_series(self, target_col='销售额', group_col='产品类别'):
        """绘制时间序列图"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        if '销售日期' not in self.data.columns:
            raise ValueError("数据中没有销售日期列")
        
        plt.figure(figsize=(15, 8))
        
        if group_col in self.data.columns:
            # 按产品类别分组绘制
            for category in self.data[group_col].unique():
                category_data = self.data[self.data[group_col] == category]
                plt.plot(category_data['销售日期'], category_data[target_col], 
                         marker='o', label=f'{group_col}: {category}')
        else:
            # 绘制整体趋势
            plt.plot(self.data['销售日期'], self.data[target_col], marker='o')
        
        plt.title(f'{target_col}的时间序列趋势', fontsize=16)
        plt.xlabel('销售日期')
        plt.ylabel(target_col)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        
        output_path = os.path.join(self.config.VISUALIZATIONS_DIR, f'{target_col}_time_series.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"时间序列图已保存到: {output_path}")
    
    def plot_categorical_comparison(self, categorical_col='产品类别', target_col='利润'):
        """绘制类别型特征的比较图"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        plt.figure(figsize=(12, 8))
        
        # 箱线图
        plt.subplot(1, 2, 1)
        sns.boxplot(x=categorical_col, y=target_col, data=self.data, palette='Set2')
        plt.title(f'{target_col}按{categorical_col}分布（箱线图）')
        plt.xlabel(categorical_col)
        plt.ylabel(target_col)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 柱状图（均值）
        plt.subplot(1, 2, 2)
        mean_values = self.data.groupby(categorical_col)[target_col].mean().sort_values(ascending=False)
        sns.barplot(x=mean_values.index, y=mean_values.values, palette='Set2')
        plt.title(f'{target_col}按{categorical_col}均值分布（柱状图）')
        plt.xlabel(categorical_col)
        plt.ylabel(f'平均{target_col}')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.config.VISUALIZATIONS_DIR, f'{target_col}_by_{categorical_col}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"类别型特征比较图已保存到: {output_path}")
    
    def plot_scatter_matrix(self, cols=None):
        """绘制散点矩阵"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        cols = cols or ['价格', '销量', '成本', '销售额', '利润', '利润率']
        cols = [col for col in cols if col in self.data.columns]
        
        # 最多选择6个特征
        cols = cols[:6]
        
        sns.pairplot(self.data[cols], diag_kind='kde', corner=True, palette='husl')
        plt.suptitle('特征散点矩阵', y=1.02, fontsize=16)
        
        output_path = os.path.join(self.config.VISUALIZATIONS_DIR, 'scatter_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"散点矩阵图已保存到: {output_path}")
    
    def plot_promotion_effect(self):
        """绘制促销活动效果分析"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        if '促销活动' not in self.data.columns:
            raise ValueError("数据中没有促销活动列")
        
        plt.figure(figsize=(15, 10))
        
        # 促销活动对销量的影响
        plt.subplot(2, 2, 1)
        sns.boxplot(x='促销活动', y='销量', data=self.data, palette='Set1')
        plt.title('促销活动对销量的影响')
        plt.grid(True, alpha=0.3)
        
        # 促销活动对利润的影响
        plt.subplot(2, 2, 2)
        sns.boxplot(x='促销活动', y='利润', data=self.data, palette='Set1')
        plt.title('促销活动对利润的影响')
        plt.grid(True, alpha=0.3)
        
        # 促销活动对利润率的影响
        plt.subplot(2, 2, 3)
        sns.boxplot(x='促销活动', y='利润率', data=self.data, palette='Set1')
        plt.title('促销活动对利润率的影响')
        plt.grid(True, alpha=0.3)
        
        # 按产品类别分析促销效果
        plt.subplot(2, 2, 4)
        sns.barplot(x='产品类别', y='利润', hue='促销活动', data=self.data, palette='Set1')
        plt.title('各产品类别促销活动效果对比')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.config.VISUALIZATIONS_DIR, 'promotion_effect.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"促销活动效果分析图已保存到: {output_path}")
    
    def plot_geographic_analysis(self):
        """绘制地区销售分析"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        if '地区' not in self.data.columns:
            raise ValueError("数据中没有地区列")
        
        plt.figure(figsize=(15, 8))
        
        # 各地区销售额对比
        region_sales = self.data.groupby('地区')['销售额'].sum().sort_values(ascending=False)
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=region_sales.index, y=region_sales.values, palette='Set3')
        plt.title('各地区销售额对比')
        plt.xlabel('地区')
        plt.ylabel('销售额')
        plt.grid(True, alpha=0.3)
        
        # 各地区产品类别销售额分布
        plt.subplot(1, 2, 2)
        region_category_sales = self.data.groupby(['地区', '产品类别'])['销售额'].sum().unstack()
        region_category_sales.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('各地区产品类别销售额分布')
        plt.xlabel('地区')
        plt.ylabel('销售额')
        plt.legend(title='产品类别')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.config.VISUALIZATIONS_DIR, 'geographic_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"地区销售分析图已保存到: {output_path}")
    
    def plot_feature_importance(self):
        """绘制特征重要性图"""
        if self.features_data is None:
            raise ValueError("请先加载特征工程后的数据")
        
        from sklearn.ensemble import RandomForestRegressor
        
        # 分离特征和目标
        X = self.features_data.drop('利润', axis=1)
        y = self.features_data['利润']
        
        # 训练随机森林模型
        rf = RandomForestRegressor(n_estimators=100, random_state=self.config.RANDOM_STATE)
        rf.fit(X, y)
        
        # 获取特征重要性
        feature_importance = pd.DataFrame({
            '特征': X.columns,
            '重要性': rf.feature_importances_
        }).sort_values(by='重要性', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='重要性', y='特征', data=feature_importance, palette='viridis')
        plt.title('特征重要性排序')
        plt.xlabel('重要性得分')
        plt.ylabel('特征')
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.config.VISUALIZATIONS_DIR, 'feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"特征重要性图已保存到: {output_path}")
    
    def run_all_visualizations(self):
        """运行所有可视化"""
        print("=== 开始数据可视化流程 ===")
        
        # 加载数据
        self.load_data()
        
        # 1. 数值型特征分布
        self.plot_numerical_distribution()
        
        # 2. 相关性热力图
        self.plot_correlation_heatmap()
        
        # 3. 时间序列分析
        self.plot_time_series(target_col='销售额')
        self.plot_time_series(target_col='利润')
        
        # 4. 类别型特征比较
        self.plot_categorical_comparison(categorical_col='产品类别', target_col='利润')
        self.plot_categorical_comparison(categorical_col='地区', target_col='利润')
        
        # 5. 散点矩阵
        self.plot_scatter_matrix()
        
        # 6. 促销活动效果分析
        self.plot_promotion_effect()
        
        # 7. 地区销售分析
        self.plot_geographic_analysis()
        
        # 8. 特征重要性
        if self.features_data is not None:
            self.plot_feature_importance()
        
        print("\n=== 数据可视化流程完成 ===")

if __name__ == "__main__":
    visualizer = DataVisualizer()
    visualizer.run_all_visualizations()
