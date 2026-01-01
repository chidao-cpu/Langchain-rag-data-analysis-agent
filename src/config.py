import os

class Config:
    # 数据配置
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    RAW_DATA_FILE = os.path.join(DATA_DIR, 'raw_data.csv')
    PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'processed_data.csv')
    FEATURES_DATA_FILE = os.path.join(DATA_DIR, 'features_data.csv')
    
    # 模型配置
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    MODEL_FILE = os.path.join(MODELS_DIR, 'best_model.pkl')
    
    # 结果配置
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    VISUALIZATIONS_DIR = os.path.join(RESULTS_DIR, 'visualizations')
    ANALYSIS_REPORT = os.path.join(RESULTS_DIR, 'analysis_report.md')
    
    # 预处理配置
    MISSING_VALUE_THRESHOLD = 0.3
    OUTLIER_METHOD = 'iqr'
    OUTLIER_THRESHOLD = 1.5
    
    # 特征工程配置
    FEATURE_SELECTION_METHOD = 'rfecv'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # 实证分析配置
    SIGNIFICANCE_LEVEL = 0.05
    MODEL_EVAL_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # 预案性分析配置
    SCENARIO_PARAMS = {
        'sales_increase': [10, 20, 30],
        'price_change': [-5, 0, 5],
        'cost_reduction': [5, 10, 15]
    }
    
    # 向量库配置
    VECTOR_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vector_db')
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    VECTOR_DB_NAME = 'analysis_reports'

# 创建必要的目录
os.makedirs(Config.VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(Config.VECTOR_DB_DIR, exist_ok=True)
