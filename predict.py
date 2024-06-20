import pandas as pd 
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, median_absolute_error
from timeseriesutil import TimeSeriesDiff, TimeSeriesEmbedder, ColumnExtractor
import akshare as ak
import matplotlib.pyplot as plt 
import matplotlib 

matplotlib.style.use('ggplot')
plt.xticks(rotation=70)

stock_data = ak.stock_zh_a_minute(symbol="sz000001",  period='1',  adjust="qfq")

data_six_columns = stock_data.iloc[:, :6]

# 保存前六列到CSV文件
csv_file_path = 'sz000001.csv'
data_six_columns.to_csv(csv_file_path, index=False)


data = pd.read_csv("sz000001.csv",
                   index_col = 0)

# 选择用于预测的特征（以日期为基础进行编码）
y = data["close"].diff() / data["close"].shift()

y[np.isnan(y)]=0

n_total = data.shape[0]
n_train = int(np.ceil(n_total*0.75))

data_train = data[:n_train]
data_test  = data[n_train:]

y_train = y[10:n_train]
y_test  = y[(n_train+10):]

# 利用Pipeline实现建模

pipeline = Pipeline([("ColumnEx", ColumnExtractor("close")),
                     ("Diff", TimeSeriesDiff()),
                     ("Embed", TimeSeriesEmbedder(10)),
                     ("Imputer", SimpleImputer()),
                     ("LinReg", LinearRegression())])
                    
pipeline.fit(data_train, y_train)
y_pred = pipeline.predict(data_test)


print('pipeline1 r2_score:',r2_score(y_test, y_pred))
# 中值绝对误差回归损失。中值绝对误差输出为非负浮点。最佳值为 0.0。
print('pipeline1 median_absolute_error:',median_absolute_error(y_test, y_pred))

cc = np.sign(y_pred)*y_test
cumulative_return = (cc+1).cumprod()
cumulative_return.plot(style="r-", rot=10)
plt.savefig("plots/累积回报率曲线1.png")
# plt.show()

"""更复杂的Pipeline
将成交量也纳入考虑，所以需要进行多个pipeline的融合。
同时引入多远交互项，以考虑非线性相关关系。
"""

pipeline_closing_price = Pipeline([("ColumnEx", ColumnExtractor("close")),
                                   ("Diff", TimeSeriesDiff()),
                                   ("Embed", TimeSeriesEmbedder(10)),
                                   ("Imputer", SimpleImputer()),
                                   ("Scaler", StandardScaler())])

pipeline_volume = Pipeline([("ColumnEx", ColumnExtractor("volume")),
                            ("Diff", TimeSeriesDiff()),
                            ("Embed", TimeSeriesEmbedder(10)),
                            ("Imputer", SimpleImputer()),
                            ("Scaler", StandardScaler())])

merged_features = FeatureUnion([("ClosingPriceFeature", pipeline_closing_price),
                                ("VolumeFeature", pipeline_volume)])

pipeline_2 = Pipeline([("MergedFeatures", merged_features),
                       ("PolyFeature",PolynomialFeatures()),
                       ("LinReg", LinearRegression())])
pipeline_2.fit(data_train, y_train)

y_pred_2 = pipeline_2.predict(data_test)

print('pipeline2 r2_score:',r2_score(y_test, y_pred_2))
print('pipeline2 median_absolute_error:',median_absolute_error(y_test, y_pred_2))

cc_2 = np.sign(y_pred_2)*y_test
cumulative_return_2 = (cc_2+1).cumprod()
cumulative_return_2.plot(style="k-", rot=10)
plt.savefig("plots/累积回报率曲线2.png")
