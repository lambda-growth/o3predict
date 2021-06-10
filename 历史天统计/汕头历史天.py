import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
file_dir = '../data'
all_csv_list = os.listdir(file_dir)
for single_csv in all_csv_list:
    single_data_frame = pd.read_csv(os.path.join(file_dir, single_csv))
    if single_csv == all_csv_list[0]:
        all_data_frame = single_data_frame
    else:
        all_data_frame = pd.concat([all_data_frame, single_data_frame], ignore_index=True)
print(all_data_frame.columns)
all_data_frame.dropna(subset=['temp_mean','dswrf_max','humi_max', 'apcp_max'], inplace=True)


def pressure_func(x):
    if x <= 50:
        return '低气压'
    else:
        return '高气压'


def dswrf_func(x):
    if x <= 250:
        return '低紫外'
    elif 250 < x <= 500:
        return '中紫外'
    elif 500 < x <= 750:
        return '高紫外'
    else:
        return '强紫外'


def humi_func(x):
    if x <= 60:
        return '低湿'
    else:
        return '高湿'


def temp_func(x):
    if x <= 10:
        return '低温'
    elif x <= 25:
        return '中温'
    else:
        return '高温'


def windlevel_func(x):
    if x <= 1:
        return '低风'
    elif x <= 3:
        return '中风'
    else:
        return '高风'


def func(x):
    if x <= 50:
        return '优'
    elif x <= 100:
        return '良'
    elif x <= 150:
        return '轻度污染'
    elif x <= 200:
        return '中度污染'
    elif x <= 300:
        return '重度污染'
    else:
        return '严重污染'

# 'temp_mean','dswrf_max','humi_max', 'apcp_max'

# plt.figure(figsize=(6, 4))
# for i, fea in enumerate(['temp_mean','dswrf_max','humi_max', 'apcp_max' ]):
#     plt.subplot(2, 2, i+1)
#     sns.distplot(all_data_frame[fea])
# plt.show()

def data_map(all_data_frame):
    all_data_frame['humi_rank'] = all_data_frame['humi_max'] .map(humi_func)
    all_data_frame['dswrf_rank'] = all_data_frame['dswrf_max'] .map(dswrf_func)
    all_data_frame['pressure_rank'] = all_data_frame['apcp_max'].map(pressure_func)
    all_data_frame['temp_rank'] = all_data_frame['temp_mean'].map(temp_func)
    all_data_frame['aqi_rank'] = all_data_frame['aqi'].map(func)
    all_data_frame = all_data_frame.join(pd.get_dummies(all_data_frame['aqi_rank']))
    all_data_frame['label'] = all_data_frame['humi_rank'] + all_data_frame['dswrf_rank'] + all_data_frame['temp_rank'] + all_data_frame['pressure_rank']
    return all_data_frame
all_data_frame = data_map(all_data_frame)

print(all_data_frame['aqi_rank'].value_counts())
res = pd.pivot_table(all_data_frame,index=['humi_rank','dswrf_rank','temp_rank','label'], values=['优', '良', '轻度污染', '中度污染', '重度污染'],aggfunc=np.sum)


city = '汕头'
test_data = pd.read_csv('../data/2020' + city + '日.csv').iloc[-493:, :]
test_data = test_data[test_data['aqi']>=80]
test_data = data_map(test_data)
test_result = pd.merge(test_data, res, how='left', on='label')
print(test_result)