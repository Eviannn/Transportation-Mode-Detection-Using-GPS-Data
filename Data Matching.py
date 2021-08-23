############################################################################################
# 作者：戴晚锐
# 项目：毕业设计数据匹配（代码1）
############################################################################################

import pandas as pd
import csv
import time
from os import listdir

start_time = time.time()
print("开始处理数据")

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


# 创建输出文件
f = open('data.csv', 'w', encoding='utf-8', newline="")
headers = ['People_Num', 'Time', 'Travel Start Time',
           'Travel End Time', 'Lat', 'Lon', 'Alt', 'Transportation Mode']
csv_writer = csv.writer(f)
csv_writer.writerow(headers)

# 读取数据
Data_files = listdir('D:\\final year project code\\Data')

for m in range(len(Data_files)):
    Trajectory_files = listdir('D:\\final year project code\\Data\\' + Data_files[m] + '\\Trajectory')
    print(end='\n')
    print('正在处理' + Data_files[m] + '文件夹')
    origin_i = 0
    for n in range(len(Trajectory_files)):
        print('\r', '文件处理进程：' + str(round((n + 1) / len(Trajectory_files) * 100, 2)) + ' %', end='', flush=True)
        labels = pd.read_table('D:\\final year project code\\Data\\' + Data_files[m] + '\\labels.txt', sep='\t')
        Trajectory = pd.read_table('D:\\final year project code\\Data\\' + Data_files[m] +
                                   '\\Trajectory\\' + Trajectory_files[n], header=None, sep=',', skiprows=6)
        size_labels = len(labels)
        size_Trajectory = len(Trajectory)

        ############################################################################################
        # labels数据预处理
        ############################################################################################

        # 将时间数据转换为时间格式
        labels['Start Time'] = pd.to_datetime(labels['Start Time'])
        labels['End Time'] = pd.to_datetime(labels['End Time'])

        # 合并时间上连续的出行段
        count = 0
        i = 0
        while i < size_labels - 1:
            if (labels['Start Time'][i + 1] - labels['End Time'][i]).seconds == 1 \
                    and labels['Transportation Mode'][i] == labels['Transportation Mode'][i + 1]:
                labels['End Time'].at[i] = labels['End Time'].at[i + 1]
                labels = labels.drop(labels.index[i + 1]).reset_index(drop=True)
                count = count + 1
                i = i - 1
            i = i + 1
            if i == size_labels - count - 1:
                break

        size_labels = len(labels)

        ############################################################################################
        # Trajectory数据预处理
        ############################################################################################
        Trajectory['Time'] = Trajectory[5] + ' ' + Trajectory[6]
        Trajectory = Trajectory.drop(columns=[2, 4, 5, 6], axis=1)
        Trajectory.columns = ['Lat', 'Lon', 'Alt', 'Time']
        Trajectory['Time'] = pd.to_datetime(Trajectory['Time'])

        ############################################################################################
        # Trajectory数据与labels数据匹配
        ############################################################################################

        origin_j = 0
        for i in range(origin_i, size_labels):
            for j in range(origin_j, size_Trajectory):
                if labels['Start Time'][i] <= Trajectory['Time'][j] <= labels['End Time'][i]:
                    csv_writer.writerow([Data_files[m], Trajectory['Time'][j], labels['Start Time'][i],
                                         labels['End Time'][i], Trajectory['Lat'][j], Trajectory['Lon'][j],
                                         Trajectory['Alt'][j], labels['Transportation Mode'][i]])
                    origin_j = j
                    origin_i = i


f.close()

end_time = time.time()

print(end='\n')
print("数据处理完毕，用时%.2f秒" % (end_time - start_time))
