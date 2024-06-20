import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import mplfinance as mpf


# 读取数据
data = pd.read_csv("sz000001.csv")

# 确保日期列为datetime类型
data['day'] = pd.to_datetime(data['day'])

# 数据概览
#print(data.describe())
# 去除第一列'day'
data_without_day = data.drop(data.columns[0], axis=1)

# 打印去除第一列后的统计概览
print(data_without_day.describe())

# 计算并打印每列的相关系数矩阵
correlation_matrix = data_without_day.corr()
print(correlation_matrix)

# 使用Seaborn热力图展示相关系数矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix of Stock Data')
plt.tight_layout()
plt.savefig("plots/相关系数矩阵图.png")
# plt.show()

data.set_index('day', inplace=True)
# 计算移动平均线
data['MA5'] = data['close'].rolling(window=5).mean()
data['MA10'] = data['close'].rolling(window=10).mean()

# 创建附加图表（移动平均线）
apds = [mpf.make_addplot(data['MA5'], color='r'),
        mpf.make_addplot(data['MA10'], color='g')]

# 绘制K线图并保存到文件
mpf.plot(data, type='line', addplot=apds, volume=True,
         figscale=1.2, figsize=(12, 8), title='sz000001',
         savefig='plots/k线图.png')
