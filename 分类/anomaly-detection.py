import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


#特征工程函数，主要是历史天的统计量
def get_mov_avg_std(df, col, N):
    """
    Given a dataframe, get mean and std dev at timestep t using values from t-1, t-2, ..., t-N.
    Inputs
        df         : dataframe. Can be of any length.
        col        : name of the column you want to calculate mean and std dev
        N          : get mean and std dev at timestep t using values from t-1, t-2, ..., t-N
    Outputs
        df_out     : same as df but with additional column containing mean and std dev
    """
    mean_list = df[col].rolling(window=N, min_periods=1).mean()
    std_list = df[col].rolling(window=N, min_periods=1).std()
    skew_list = df[col].rolling(window=N, min_periods=1).skew()
    median_list = df[col].rolling(window=N, min_periods=1).median()
    min_list = df[col].rolling(window=N, min_periods=1).min()
    max_list = df[col].rolling(window=N, min_periods=1).max()
    q10_list = df[col].rolling(window=N, min_periods=1).quantile(0.1)
    q50_list = df[col].rolling(window=N, min_periods=1).quantile(0.5)
    q90_list = df[col].rolling(window=N, min_periods=1).quantile(0.9)
    # Add one timestep to the predictions ,这里又shift了一步
    mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
    std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))
    skew_list = np.concatenate((np.array([np.nan]), np.array(skew_list[:-1])))
    median_list = np.concatenate((np.array([np.nan]), np.array(median_list[:-1])))
    min_list = np.concatenate((np.array([np.nan]), np.array(min_list[:-1])))
    max_list = np.concatenate((np.array([np.nan]), np.array(max_list[:-1])))
    q10_list = np.concatenate((np.array([np.nan]), np.array(q10_list[:-1])))
    q50_list = np.concatenate((np.array([np.nan]), np.array(q50_list[:-1])))
    q90_list = np.concatenate((np.array([np.nan]), np.array(q90_list[:-1])))
    # Append mean_list to df
    df_out = df.copy()
    df_out[col + str(N)+'_mean'] = mean_list
    df_out[col + str(N) + '_skew'] = skew_list
    df_out[col + str(N) + '_median'] = median_list
    df_out[col + str(N)+'_std'] = std_list
    df_out[col + str(N) + '_max'] = max_list
    df_out[col + str(N) + '_min'] = min_list
    # df_out[col + str(N)+'_q10'] = q10_list
    # df_out[col + str(N) + '_q50'] = q50_list
    # df_out[col + str(N) + '_q90'] = q90_list
    return df_out


#读取数据
data = pd.read_csv('../data/2020汕头日.csv')

# 获取类别标签，aqi < 100为类别1， 大于100为类别2，target字段
def func(x):
    if x <= 100:
        return 1
    else:
        return -1

data['target'] = data['aqi'].map(func)

#选择特征，包括六项，风向，温度、气压、紫外线等特征
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


#需要对数据进行上采样
from imblearn.over_sampling import SMOTE
from collections import Counter
# 定义SMOTE模型，random_state相当于随机数种子的作用
# smo = SMOTE(random_state=42)
# X_smo, y_smo = smo.fit_sample(X, Y)
# print(Counter(y_smo))
# print(X_smo, y_smo)

# 正样本和负样本分开，LOF、IF模型采用的是半监督学习，假设只有正样本，用正样本训练，测试集包含正负样本，这里按1:1比例划分
data_pos = data[data['target'] == 1]
data_neg = data[data['target'] == -1]

from sklearn.utils import shuffle

# data_pos = shuffle(data_pos)
train = data_pos.iloc[:-50,:]
test = data_pos.iloc[-50:,:]
test = pd.concat([test,data_neg])


X = train[train.columns[:-1]]
y = train['target']

X_test = test[test.columns[:-1]]
y_test = test['target']
print(X.shape)
print(y.shape)
print(y.value_counts())
#数据标准化
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
X_test = StandardScaler().fit_transform(X_test)
from sklearn.neighbors import LocalOutlierFactor
#模型定义
LOF = LocalOutlierFactor(n_neighbors=100, algorithm='auto', leaf_size=10, metric='minkowski', p=3,
                         metric_params=None, contamination=0.3, novelty=True, n_jobs=-1).fit(X)
LOF_predict = LOF.predict(X)

# from sklearn.ensemble import IsolationForest
#
# LOF = IsolationForest(n_estimators=900, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=False, n_jobs=-1,
#                       random_state=42, verbose=0, warm_start=False).fit(X)
# LOF_predict = LOF.predict(X)

#模型判定
from sklearn.metrics import confusion_matrix   #混淆矩阵
from sklearn.metrics import accuracy_score     #准确率
from sklearn.metrics import precision_score    #精确率
from sklearn.metrics import recall_score       #召回率
from sklearn.metrics import f1_score           #F1-得分
from sklearn.metrics import classification_report

print("混淆矩阵：")
print(confusion_matrix(y,LOF_predict))
print("标准化混淆矩阵：")
print(confusion_matrix(y,LOF_predict,normalize='true'))
print("得分详细：")
print(classification_report(y,LOF_predict))


LOF_test = LOF.predict(X_test)
print("混淆矩阵：")
print(confusion_matrix(y_test,LOF_test))
print("标准化混淆矩阵：")
print(confusion_matrix(y_test,LOF_test,normalize='true'))
print("得分详细：")
print(classification_report(y_test,LOF_test))


# #预测结果展示比较
# IF_predict = IF.predict(X)
# train_if = pd.DataFrame(IF_predict,columns=["predict"])
# train_if["y"] = y
# print(train_if)

# from sklearn.metrics import confusion_matrix   #混淆矩阵
# from sklearn.metrics import accuracy_score     #准确率
# from sklearn.metrics import precision_score    #精确率
# from sklearn.metrics import recall_score       #召回率
# from sklearn.metrics import f1_score           #F1-得分
# from sklearn.metrics import classification_report
#
# print("混淆矩阵：")
# print(confusion_matrix(y,IF_predict))
# print("标准化混淆矩阵：")
# print(confusion_matrix(y,IF_predict,normalize='true'))
# print("得分详细：")
# print(classification_report(y,IF_predict))
# print(train_if[train_if['predict']==1].index)


