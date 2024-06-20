import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# 将时间序列数据转换为适合机器学习模型输入的格式
def embed_time_series(x, k):
    n = len(x)

    if k >= n: 
        raise "Can not deal with k greater than the length of x" 
    
    output_x = list(map(lambda i: list(x[i:(i+k)]), 
                        range(0, n-k)))
    return np.array(output_x)

# 使用embed_time_series函数将时间序列数据进行转换,将一维时间序列数据转换为高维向量表示
class TimeSeriesEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, k):
        self.k = k 
    def fit(self, X, y= None):
        return self
    def transform(self, X, y = None):
        return embed_time_series(X, self.k)

# 从数据中提取指定的列
class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.column_name]

# 计算时间序列数据的差分
class TimeSeriesDiff(BaseEstimator, TransformerMixin):
    def __init__(self, k=1):
        self.k = k 
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if type(X) is pd.core.frame.DataFrame or type(X) is pd.core.series.Series:
            return X.diff(self.k) / X.shift(self.k)
        else:
            raise "Have to be a pandas data frame or Series object!"
