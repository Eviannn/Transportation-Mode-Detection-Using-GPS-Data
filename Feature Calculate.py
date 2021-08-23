############################################################################################
# 作者：戴晚锐
# 项目：毕业设计数据特征计算（代码3）
############################################################################################

import pandas as pd
import time
from geopy.distance import geodesic

start_time = time.time()
print("开始处理数据")

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


data = pd.read_csv('D:\\final year project code\\Trajectory and mode(after data process).csv')

time_gap = []
distance_gap = []
speed = []
acceleration = []
total_time = []
total_time_count = 0
total_distance = []
total_distance_count = 0

# 将时间数据转换为时间格式
data['Time'] = pd.to_datetime(data['Time'])

# 控制时间频次
i = 0
count = []

while i < len(data)-1:
    if data['Travel Count'][i] == data['Travel Count'][i+1] \
            and (data['Time'][i+1] - data['Time'][i]).seconds < 50:
        count.append(i+1)
        data['Time'].at[i+1] = data['Time'].at[i]
    i = i + 1
data = data.drop(count).reset_index(drop=True)

# 计算两点之间的时间间隔与距离
for i in range(len(data)-1):
    if data['Travel Count'][i] == data['Travel Count'][i+1]:
        time_gap.append((data['Time'][i + 1] - data['Time'][i]).seconds)
        distance_gap.append(geodesic((data['Lat'][i], data['Lon'][i]), (data['Lat'][i + 1], data['Lon'][i + 1])).m)
    else:
        time_gap.append('N.A')
        distance_gap.append('N.A')

time_gap.append('N.A')
distance_gap.append('N.A')

# 计算两点之间的速度
for i in range(len(data)):
    if time_gap[i] != 'N.A':
        speed.append(round(distance_gap[i]/time_gap[i], 2))
    else:
        speed.append('N.A')

# 计算两点之间的加速度
for i in range(len(data)-1):
    if speed[i] != 'N.A' and speed[i+1] != 'N.A':
        acceleration.append(round((speed[i+1]-speed[i])/time_gap[i], 2))
    else:
        acceleration.append('N.A')

acceleration.append('N.A')

# 计算两段行程的总时间和总距离
for i in range(len(data)):
    if time_gap[i] != 'N.A':
        total_time_count = total_time_count + time_gap[i]
        total_time.append('N.A')
        total_distance_count = total_distance_count + distance_gap[i]
        total_distance.append('N.A')
    else:
        total_time.append(total_time_count)
        total_distance.append(total_distance_count)
        total_time_count = 0
        total_distance_count = 0

data['Time Gap(s)'] = time_gap
data['Distance(m)'] = distance_gap
data['Speed(m/s)'] = speed
data['Acceleration(m/s^2)'] = acceleration
data['Total Time(s)'] = total_time
data['Total Distance(m)'] = total_distance


data.to_csv('D:\\final year project code\\Trajectory and mode(after feature calculate).csv', index=0)

end_time = time.time()

print("数据处理完毕，用时%.2f秒" % (end_time - start_time))