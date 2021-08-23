############################################################################################
# 作者：戴晚锐
# 项目：毕业设计数据处理（代码2）
############################################################################################

import pandas as pd
import time


start_time = time.time()
print("开始处理数据")

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

travel_count = 1

# 读取数据
data = pd.read_csv('D:\\final year project code\\Trajectory and mode(after data matching).csv')
data['Travel Count'] = 1
data['Time'] = pd.to_datetime(data['Time'])
count = []

# 对于每一个轨迹数据的时间点，若对应多条交通出行方式数据，则全部删除
for i in range(len(data) - 1):
    if (data['Time'][i + 1] - data['Time'][i]).seconds == 0:
        count.append(i)
        count.append(i+1)

data = data.drop(count).reset_index(drop=True)

count.clear()

# 删除经纬度异常的数据
for i in range(len(data)):
    if data['Lat'][i] > 90 or data['Lat'][i] < 0 or data['Lon'][i] > 180 or data['Lon'][i] < -180:
        count.append(i)

data = data.drop(count).reset_index(drop=True)

# 分割行程段
for i in range(len(data)):
    data['Travel Count'].at[i] = travel_count
    if i != len(data)-1:
        if data['Travel Start Time'][i] != data['Travel Start Time'][i+1] or \
                (data['Time'][i + 1] - data['Time'][i]).seconds > 1800:
            travel_count = travel_count + 1

result = pd.value_counts(data['Travel Count'])
print(result)

data.to_csv('D:\\final year project code\\Trajectory and mode(after data process).csv', index=0)

end_time = time.time()

print(end='\n')
print("数据处理完毕，用时%.2f秒" % (end_time - start_time))

