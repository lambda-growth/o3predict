# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import pandas as pd


def get_data(data):


    # 获取类别标签
    def func(x):
        if x <= 100:
            return 1
        else:
            return -1

    data['target'] = data['aqi'].map(func)
    data = data.loc[:,['aqi','o3_8h','wda_max', 'wda_mean', 'wda_min', 'temp_max',
           'temp_mean', 'temp_min', 'dswrf_max', 'dswrf_mean', 'dswrf_min',
           'humi_max', 'humi_mean', 'humi_min', 'apcp_max', 'apcp_mean',
           'apcp_min', 'target']]
    #获取特征的历史平均，最大最小值,'dswrf_max','humi_max','apcp_max'
    # for i in [3, 7, 14]:
    #     for j in ['wda_max','temp_max','dswrf_max','humi_max']:
    #         data = get_mov_avg_std(data,j,i)

    # for k in [1, 2, 3]:
    #     data['aqi_shift' + str(k)] = data['aqi'].shift(k)
    #     data['o3_shift' + str(k)] = data['o3_8h'].shift(k)
    #     data['dswrf_max' + str(k)] = data['dswrf_max'].shift(k)

    data.fillna(method='ffill',inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.dropna(axis=0, inplace=True)
    X = data.drop(['target'], axis=1)
    Y = data['target']
    return X,Y

def get_result(Y, y_pred):
    from sklearn.metrics import confusion_matrix  # 混淆矩阵
    from sklearn.metrics import accuracy_score  # 准确率
    from sklearn.metrics import precision_score  # 精确率
    from sklearn.metrics import recall_score  # 召回率
    from sklearn.metrics import f1_score  # F1-得分
    from sklearn.metrics import classification_report
    print("混淆矩阵：")
    print(confusion_matrix(Y, y_pred))
    print("标准化混淆矩阵：")
    print(confusion_matrix(Y, y_pred, normalize='true'))
    print("得分详细：")
    print(classification_report(Y, y_pred))
# from imblearn.over_sampling import SMOTE
from collections import Counter
# # 定义SMOTE模型，random_state相当于随机数种子的作用
# smo = SMOTE(random_state=42)
# X_smo, y_smo = smo.fit_sample(X, Y)
# print(Counter(y_smo))
# print(X_smo, y_smo)


data = pd.read_csv('../data/2020揭阳日.csv')
X,Y = get_data(data)
from imblearn.over_sampling import SMOTE

# 定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(random_state=42)
X, Y = smo.fit_sample(X, Y)
print(Counter(Y))


# fit the model
outliers_fraction = 0.1
# clf = LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction, novelty=True).fit(X)
clf = LocalOutlierFactor(n_neighbors=100, algorithm='auto', leaf_size=10, metric='minkowski', p=3,
                         metric_params=None, contamination=0.3, novelty=True, n_jobs=-1).fit(X)

y_pred = clf.predict(X)
scores_pred = clf.negative_outlier_factor_
threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)  # 根据异常样本比例，得到阈值，用于绘图
get_result(Y, y_pred)


data = pd.read_csv('../data/2020汕头日.csv')
X_test,Y_test = get_data(data)
test_pred = clf.predict(X_test)
get_result(Y_test, test_pred)