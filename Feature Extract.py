############################################################################################
# 作者：戴晚锐
# 项目：毕业设计数据特征提取（代码4）
############################################################################################

import pandas as pd
import numpy as np
import time

start_time = time.time()
print("开始处理数据")

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


# 读取源文件
data = pd.read_csv('D:\\final year project code\\Trajectory and mode(after feature calculate).csv')

# 设置输出文件列名
data_feature = pd.DataFrame(columns=['Travel Count', 'Transportation Mode', 'Max Speed(m/s)', '95% Speed(m/s)',
                                     '75% Speed(m/s)', 'Mean Speed(m/s)', 'Speed Std', 'Max Acceleration(m/s^2)',
                                     '95% Acceleration(m/s^2)', '75% Acceleration(m/s^2)',
                                     'Mean Acceleration(m/s^2)', 'Acceleration Std', 'Non 0 Mean Speed(m/s)',
                                     'Non 0 Mean Acceleration(m/s^2)', 'Total Time(s)', 'Total Distance(m)'])

############################################################################################
# 删除异常速度数据
############################################################################################

count_speed = []

for i in range(len(data)):
    if data['Transportation Mode'][i] == 'train' or data['Transportation Mode'][i] == 'subway':
        if data['Speed(m/s)'][i] != 'N.A':
            if float(data['Speed(m/s)'][i]) > 100:
                count_speed.append(i)
    if data['Transportation Mode'][i] == 'taxi' or data['Transportation Mode'][i] == 'bus' \
            or data['Transportation Mode'][i] == 'car':
        if data['Speed(m/s)'][i] != 'N.A':
            if float(data['Speed(m/s)'][i]) > 45:
                count_speed.append(i)
    if data['Transportation Mode'][i] == 'walk':
        if data['Speed(m/s)'][i] != 'N.A':
            if float(data['Speed(m/s)'][i]) > 5:
                count_speed.append(i)
    if data['Transportation Mode'][i] == 'bike':
        if data['Speed(m/s)'][i] != 'N.A':
            if float(data['Speed(m/s)'][i]) > 10:
                count_speed.append(i)

data = data.drop(count_speed).reset_index(drop=True)

############################################################################################
# 删除异常加速度数据
############################################################################################
count_speed_a = []

for i in range(len(data)):
    if data['Transportation Mode'][i] == 'train' or data['Transportation Mode'][i] == 'subway':
        if data['Acceleration(m/s^2)'][i] != 'N.A':
            if float(data['Acceleration(m/s^2)'][i]) > 8:
                count_speed_a.append(i)
    if data['Transportation Mode'][i] == 'taxi' or data['Transportation Mode'][i] == 'bus' \
            or data['Transportation Mode'][i] == 'car':
        if data['Acceleration(m/s^2)'][i] != 'N.A':
            if float(data['Acceleration(m/s^2)'][i]) > 12:
                count_speed_a.append(i)
    if data['Transportation Mode'][i] == 'walk':
        if data['Acceleration(m/s^2)'][i] != 'N.A':
            if float(data['Acceleration(m/s^2)'][i]) > 3:
                count_speed_a.append(i)
    if data['Transportation Mode'][i] == 'bike':
        if data['Acceleration(m/s^2)'][i] != 'N.A':
            if float(data['Acceleration(m/s^2)'][i]) > 5:
                count_speed_a.append(i)

data = data.drop(count_speed_a).reset_index(drop=True)

############################################################################################
# 删除轨迹点数量小于4个的行程段数据
############################################################################################

count = 1
count_list = []
location = []


for i in range(len(data)-1):
    if data['Travel Count'][i] == data['Travel Count'][i+1]:
        count = count + 1
    else:
        count_list.append(count)
        count = 1
count_list.append(count)

for i in range(len(count_list)):
    if count_list[i] < 4:
        location.append(i+1)

data = data[-data['Travel Count'].isin(location)]
data = data.reset_index(drop=True)


############################################################################################
# 计算特征变量
############################################################################################

speed = []
acceleration = []
travel_count_feature = []
mode_feature = []
speed_max_feature = []
speed_95_feature = []
speed_75_feature = []
speed_mean_feature = []
speed_std_feature = []
acceleration_max_feature = []
acceleration_95_feature = []
acceleration_75_feature = []
acceleration_mean_feature = []
acceleration_std_feature = []

total_time_feature = []
total_distance_feature = []

non_0_speed = []
non_0_speed_acceleration = []
non_0_speed_mean_feature = []
non_0_speed_acceleration_mean_feature = []

for i in range(len(data)):
    if i != len(data)-1:
        if data['Travel Count'][i] == data['Travel Count'][i+1]:
            if float(data['Speed(m/s)'][i]) != 0:
                non_0_speed.append(float(data['Speed(m/s)'][i]))
                non_0_speed_acceleration.append(data['Acceleration(m/s^2)'][i])
            speed.append(float(data['Speed(m/s)'][i]))
            acceleration.append(data['Acceleration(m/s^2)'][i])
        else:
            if 'N.A' in acceleration:
                acceleration.remove('N.A')
            acceleration = list(map(float, acceleration))
            acceleration = list(map(abs, acceleration))
            speed_np = np.array(speed)
            acceleration_np = np.array(acceleration)
            travel_count_feature.append(data['Travel Count'][i])
            mode_feature.append(data['Transportation Mode'][i])

            speed_max_feature.append(max(speed))
            speed_95_feature.append(round(np.percentile(speed_np, 95), 2))
            speed_75_feature.append(round(np.percentile(speed_np, 75), 2))
            speed_mean_feature.append(round(float(np.mean(speed_np)), 2))
            speed_std_feature.append(round(float(np.std(speed_np)), 2))

            acceleration_max_feature.append(max(acceleration))
            acceleration_95_feature.append(round(np.percentile(acceleration_np, 95), 2))
            acceleration_75_feature.append(round(np.percentile(acceleration_np, 75), 2))
            acceleration_mean_feature.append(round(float(np.mean(acceleration_np)), 2))
            acceleration_std_feature.append(round(float(np.std(acceleration_np)), 2))

            total_time_feature.append(float(data['Total Time(s)'][i]))
            total_distance_feature.append(round(float(data['Total Distance(m)'][i]), 2))

            speed = []
            acceleration = []
            if 'N.A' in non_0_speed_acceleration:
                non_0_speed_acceleration.remove('N.A')
            non_0_speed_acceleration = list(map(float, non_0_speed_acceleration))
            non_0_speed_acceleration = list(map(abs, non_0_speed_acceleration))
            non_0_speed_np = np.array(non_0_speed)
            non_0_acceleration_np = np.array(non_0_speed_acceleration)
            non_0_speed_mean_feature.append(round(float(np.mean(non_0_speed_np)), 2))
            non_0_speed_acceleration_mean_feature.append(round(float(np.mean(non_0_acceleration_np)), 2))

            non_0_speed = []
            non_0_speed_acceleration = []

    else:
        if 'N.A' in acceleration:
            acceleration.remove('N.A')
        acceleration = list(map(float, acceleration))
        acceleration = list(map(abs, acceleration))
        speed_np = np.array(speed)
        acceleration_np = np.array(acceleration)
        travel_count_feature.append(data['Travel Count'][i])
        mode_feature.append(data['Transportation Mode'][i])

        speed_max_feature.append(max(speed))
        speed_95_feature.append(round(np.percentile(speed_np, 95), 2))
        speed_75_feature.append(round(np.percentile(speed_np, 75), 2))
        speed_mean_feature.append(round(float(np.mean(speed_np)), 2))
        speed_std_feature.append(round(float(np.std(speed_np)), 2))

        acceleration_max_feature.append(max(acceleration))
        acceleration_95_feature.append(round(np.percentile(acceleration_np, 95), 2))
        acceleration_75_feature.append(round(np.percentile(acceleration_np, 75), 2))
        acceleration_mean_feature.append(round(float(np.mean(acceleration_np)), 2))
        acceleration_std_feature.append(round(float(np.std(acceleration_np)), 2))

        total_time_feature.append(float(data['Total Time(s)'][i]))
        total_distance_feature.append(round(float(data['Total Distance(m)'][i]), 2))

        if 'N.A' in non_0_speed_acceleration:
            non_0_speed_acceleration.remove('N.A')
        non_0_speed_acceleration = list(map(float, non_0_speed_acceleration))
        non_0_speed_acceleration = list(map(abs, non_0_speed_acceleration))
        non_0_speed_np = np.array(non_0_speed)
        non_0_acceleration_np = np.array(non_0_speed_acceleration)
        non_0_speed_mean_feature.append(round(float(np.mean(non_0_speed_np)), 2))
        non_0_speed_acceleration_mean_feature.append(round(float(np.mean(non_0_acceleration_np)), 2))

############################################################################################
# 给列赋值
############################################################################################

data_feature['Travel Count'] = travel_count_feature
data_feature['Transportation Mode'] = mode_feature
data_feature['Max Speed(m/s)'] = speed_max_feature
data_feature['95% Speed(m/s)'] = speed_95_feature
data_feature['75% Speed(m/s)'] = speed_75_feature
data_feature['Mean Speed(m/s)'] = speed_mean_feature
data_feature['Speed Std'] = speed_std_feature
data_feature['Max Acceleration(m/s^2)'] = acceleration_max_feature
data_feature['95% Acceleration(m/s^2)'] = acceleration_95_feature
data_feature['75% Acceleration(m/s^2)'] = acceleration_75_feature
data_feature['Mean Acceleration(m/s^2)'] = acceleration_mean_feature
data_feature['Acceleration Std'] = acceleration_std_feature
data_feature['Non 0 Mean Speed(m/s)'] = non_0_speed_mean_feature
data_feature['Non 0 Mean Acceleration(m/s^2)'] = non_0_speed_acceleration_mean_feature
data_feature['Total Time(s)'] = total_time_feature
data_feature['Total Distance(m)'] = total_distance_feature

data_feature.to_excel('D:\\final year project code\\Feature.xlsx', index=0)

end_time = time.time()

print(end='\n')
print("数据处理完毕，用时%.2f秒" % (end_time - start_time))
