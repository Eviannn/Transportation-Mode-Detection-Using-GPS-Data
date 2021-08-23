############################################################################################
# 作者：戴晚锐
# 项目：毕业设计数据分析（代码5）
############################################################################################

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pandas_profiling

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

data = pd.read_excel('D:\\final year project code\\data analysis.xlsx')

profile = data.profile_report(title='Data Analysis')
profile.to_file(output_file='D:\\final year project code\\Data Analysis.html')

print(data.corr())

# 绘制相关系数矩阵
sns.heatmap(data.corr(), square=True, cmap='Purples')
sns.set(font_scale=1.5)
plt.rc('font', family='Times New Roman', size=4)
plt.show

