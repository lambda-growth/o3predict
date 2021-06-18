import argparse
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, ShuffleSplit
from datetime import datetime
from sklearn import  metrics
from sklearn.metrics import mean_squared_error, classification_report, f1_score
import numpy as np
import pandas as pd
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 站点map
def point_map():
    citys = ['广州','韶关','深圳','珠海','佛山','江门','湛江',
                    '茂名','肇庆','惠州','梅州','汕尾','汕头','阳江','河源','清远',
                    '东莞','中山','潮州','揭阳','云浮']
    point_ls = []
    for city in citys:
        data = pd.read_csv('./data/广东站点/'+ city + '.csv').iloc[:-128, :]
        point_ls.extend(data.pointname.unique())
    point_dict = dict(zip(point_ls, range(len(point_ls))))
    return point_dict
#1. 定义特征工程函数
def data_pre(df,pollution,future,point_dic):
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
    data = df.loc[:, ['aqi','co','no2','so2','o3_8h','pm2_5','pm10','pointname','time','longitude','latitude']]
    # 原特征
    data['pointname'] = data['pointname'].map(point_dic).astype(int)
    data.loc[:,'time'] =  pd.to_datetime(data['time'], format='%Y-%m-%d')
    data['month'] = data['time'].dt.month
#筛选月份

    holidays_exception = ['2017-01-01', '2017-01-02', '2017-01-27', '2017-01-28', '2017-01-29',
               '2017-01-30', '2017-01-31', '2017-02-01', '2017-02-02', '2017-04-02',
               '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
               '2017-05-28', '2017-05-29', '2017-05-30', '2017-10-01', '2017-10-02',
               '2017-10-03', '2017-10-04', '2017-10-05', '2017-10-06', '2017-10-07',
               '2017-10-08', '2017-12-30', '2017-12-31',
               '2018-01-01', '2018-02-15', '2018-02-16', '2018-02-17', '2018-02-18',
               '2018-02-19', '2018-02-20', '2018-02-21', '2018-04-05', '2018-04-06',
               '2018-04-07', '2018-04-29', '2018-04-30', '2018-05-01', '2018-06-16',
               '2018-06-17', '2018-06-18', '2018-09-22', '2018-09-23', '2018-09-24',
               '2018-10-01', '2018-10-02', '2018-10-03', '2018-10-04', '2018-10-05',
               '2018-10-06', '2018-10-07', '2018-12-30', '2018-12-31'
               '2019-01-01', '2019-02-04', '2019-02-05', '2019-02-06', '2019-02-07',
               '2019-02-08', '2019-02-09', '2019-02-10', '2019-04-05', '2019-04-06',
               '2019-04-07', '2019-04-29', '2019-04-30', '2019-05-01', '2019-06-07',
               '2019-06-08', '2019-06-09', '2019-09-13', '2019-09-14', '2019-09-15',
               '2019-10-01', '2019-10-02', '2019-10-03', '2019-10-04', '2019-10-05',
               '2019-10-06', '2019-10-07', '2019-12-30', '2019-12-31'
               '2020-01-01', '2020-01-22', '2020-01-23', '2020-01-24', '2020-01-27',
               '2020-01-28', '2020-01-29', '2020-01-30', '2020-01-31', '2020-02-01',
               '2020-04-06', '2020-05-01', '2020-05-04', '2020-05-05', '2020-06-25',
               '2020-06-26', '2020-10-01', '2020-10-02', '2020-10-05', '2020-10-06',
               '2020-10-07', '2020-10-08', '2021-01-01', '2021-02-11', '2021-02-12',
               '2021-02-15', '2021-02-16', '2021-02-17', '2021-04-05', '2021-05-03',
               '2021-05-04', '2021-05-05', '2021-06-14', '2021-09-20', '2021-09-21',
               '2021-10-01', '2021-10-04', '2021-10-05', '2021-10-06', '2021-10-07',
               ]
    workdays_exception = [
        '2017-01-22', '2017-02-04', '2017-04-01','2017-05-27','2017-09-30',
        '2018-02-11', '2018-02-24', '2018-04-08', '2018-04-28', '2018-09-29',
        '2018-09-30', '2018-12-29',
        '2019-02-02'  # 春节前调休周末上班
        '2019-02-03'
        '2019-05-05'  # 劳动节调休周末上班
        '2020-02-01',  # 春节, 周六
        '2020-04-26',  # 劳动节, 周日
        '2020-05-09',  # 劳动节, 周六
        '2020-06-28',  # 端午, 周日
        '2020-09-27',  # 国庆,周六
        '2020-10-10',  # 国庆,周六
        '2021-02-07',  # 春节前调休,周日，2021年开始
        '2021-02-20',  # 春节后调休，周六
        '2021-04-25',  # 五一调休,周日
        '2021-05-08',  # 五一调休,周六
        '2021-09-18',  # 中秋调休,周六
        '2021-09-26',  # 中秋调休,周日
        '2021-10-09',  # 国庆调休,周日
    ]

    def is_workday(day=None):
        """
            Args:
                day: 日期, 默认为今日

            Returns:
                True: 上班
                False: 放假
        """
        # 如果不传入参数则为今天
        today = datetime.today()
        # logger.info(today)
        day = day or today

        week_day = datetime.weekday(day) + 1  # 今天星期几(星期一 = 1，周日 = 7)
        is_work_day_in_week = week_day in range(1, 6)  # 这周是不是非周末，正常工作日, 不考虑调假
        day_str = day.strftime("%Y-%m-%d")

        if day_str in workdays_exception:
            return True
        elif day_str in holidays_exception:
            return False
        elif is_work_day_in_week:
            return True
        else:
            return False

    def is_holiday(day=None):
        # 如果不传入参数则为今天
        today = datetime.today()
        day = day or today
        if is_workday(day):
            return False
        else:
            return True
    data['weekday'] = pd.to_datetime(data['time']).apply(lambda x: x.weekday()+1)
    data['isholiday'] = data['time'].apply(lambda x: int(is_holiday(x)))
    to_one_hot = data['time'].dt.day_name()
    data = data.join(pd.get_dummies(to_one_hot))
    # data = data[(data['month']==6)|(data['month']==7)|(data['month']==8)]
    # data = data.join(pd.get_dummies(data[['winddirect','weather']])) #日数据中的类别特征，未来数据不含，删除
    # data.drop(columns= ['winddirect','weather'], inplace=True)
    #获取特征的历史平均，最大最小值,'dswrf_max','humi_max','apcp_max'
    for i in [3, 7, 14]:
        for j in ['aqi','co','no2','so2','o3_8h','pm2_5','pm10']:
            data = get_mov_avg_std(data,j,i)

    #获得时间特征
    data.loc[:,'time'] =  pd.to_datetime(data['time'], format='%Y-%m-%d')
    data['year'] = data['time'].dt.year
    data['month'] = data['time'].dt.month
    data.drop(['time'], axis=1,inplace=True)



    #臭氧历史特征，特征偏移
    if future != 0:
        for k in [future,future+1,future+2]:
            data['aqi_shift'+str(k)] = data['aqi'].shift(k)
            data['o3_shift' + str(k)] = data['o3_8h'].shift(k)
            # data['dswrf_max' + str(k)] = data['dswrf_max'].shift(k)

    #空值填充，删除无前值行

    data.fillna(method='ffill',inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.dropna(axis=0, inplace=True)
    #采样方法
    columns = data.columns
    data = np.array(data)
    # 前7个特征为当天的污染物特征，故去除，防止数据泄露
    X, Y = data[:-future,:].tolist(),data[future:,pollution].tolist()
    return X,Y
# 2、 模型构建，本例尝试了多种算法模型，包括MLP，随机森林，xgboost、lgbm等。
#神经网络建模MLP
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
def mlp():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((105,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(500, activation="sigmoid"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(500, activation="sigmoid"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(1, activation='relu')
    ])

    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    return model

def create_model(X_train,y_train,X_test):
    n_fold = 5
    predicted_value = np.zeros(len(X_test))
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X_train):
        x_train_, x_valid_= X_train[train_idx], X_train[val_idx]
        y_train_, y_valid_ = y_train[train_idx], y_train[val_idx]
        # 神经网络
        model = mlp()
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=100,
            verbose=0
        )
        model.fit(x_train_, y_train_,epochs=200,batch_size=128,validation_data=(x_valid_,y_valid_ ), verbose=1,
            callbacks=[early_stopping])
        # 模型预测
        valid_pred = model.predict(x_valid_)
        print('MSE: ', mean_squared_error(y_valid_, valid_pred))
        predicted_value += np.array(tf.squeeze(model.predict(X_test)))
    predicted_value /= n_fold
    return predicted_value , model

# 显示模型效果
def piture_plot(y_test, predicted_value,pointname):

    # print(len(y_test[60:]))
    y1 = list(np.array(y_test[:60]) - 15)
    y2 = list(np.array(y_test[:60]) + 15)
    plt.rcParams['figure.figsize'] = (12.0, 5.0)
    fig = plt.figure()
    # 画区间图
    ax1 = fig.add_subplot(111)
    ax1.fill_between(range(len(y_test[:60])), y1, y2, alpha=.08, color='b')
    ax1.plot(y_test[:60], color='red', label='Real aqi value')
    ax1.plot(predicted_value[:60], color='blue', label='Predicted aqi value')
    ax1.set_ylabel('AQI值', fontsize='15')
    plt.title(pointname+' future  days aqi Prediction')
    plt.xlabel('Time')
    plt.ylabel('aqi value')
    plt.legend()
    plt.savefig('aqi.png')
    # plt.show()
    plt.clf()

   # wrong_ls 计算错误分布，准确率判断依据是，在真实值正负15以内范围为标准，超过范围判定为预测错误。
    wrong_ls = []
    precision = 0
    for i in range(len(predicted_value)):
        if y_test[i] - 15 <= predicted_value[i] <= y_test[i] + 15:
            precision += 1
        else:
            wrong_ls.append(y_test[i])
    #画正确预测和错误预测的分布
    sns.distplot(np.array(wrong_ls), kde=False,label = 'false aqi distribution')
    sns.distplot(np.array(y_test),kde=False,label = 'original aqi distribution')
    plt.title('aqi distribution')
    plt.xlabel('aqi value')
    plt.ylabel('number')
    plt.legend()
    plt.savefig('distribution.png')
    # plt.show()
    plt.clf()
    print(np.array(y_test).mean())
    precision_rate = precision / len(predicted_value)
    # 返回mse，和准确率
    print(mean_squared_error(predicted_value, y_test))
    print('准确率: %.6f' % precision_rate)
    return precision_rate


def train():
    X,Y = [],[]
    predict_map = {'aqi': 0, 'co': 1, 'no2': 2, 'so2': 3, 'o3_8h': 4, 'pm2_5': 5, 'pm10': 6}
    # 参数设定： 预测的污染物类型，预测的未来第几天
    pollution = 'aqi'
    future = 1
    # 数据读取
    data = pd.read_csv('./data/广东站点/merge.csv')
    data['pointname'] = data['cityname']+data['pointname']
    point_ls = data['pointname'].unique()
    point_dic = dict(zip(point_ls, range(len(point_ls))))
    anomaly_point = []
    for point in point_dic.keys():
        point_data = data[data['pointname']==point][:-100]
        if point_data['aqi'].mean() <=10:
            anomaly_point.append(point)
            continue
        train_X, train_Y = data_pre(point_data,predict_map[pollution],future,point_dic)

        X.extend(train_X)
        Y.extend(train_Y)

    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    from collections import Counter
    # 数据划分，训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    import lightgbm as lgb
    model = lgb.LGBMRegressor(objective='regression', num_leaves=70,
                      learning_rate=0.05, n_estimators=1400,
                      max_depth=100,
                      min_child_samples=19,
                      min_child_weight=0.001,
                      metric='rmse', reg_alpha=0.3,
                      reg_lambda=0.5,
                      bagging_fraction=0.6,
                      feature_fraction=0.56)
    model.fit(X, Y)
    pickle.dump(model, open('LGBboost_pointaqi_Regressor' + str(future) + '.dat', 'wb'))
    # predicted_value = model.predict(X_test)
    # piture_plot(y_test, predicted_value)
def test():
    X, Y = [], []
    predict_map = {'aqi': 0, 'co': 1, 'no2': 2, 'so2': 3, 'o3_8h': 4, 'pm2_5': 5, 'pm10': 6}
    # 参数设定： 预测的污染物类型，预测的未来第几天
    pollution = 'aqi'
    future = 1
    # 数据读取
    data = pd.read_csv('./data/广东站点/merge.csv')
    data['pointname'] = data['cityname'] + data['pointname']
    point_ls = data['pointname'].unique()
    point_dic = dict(zip(point_ls, range(len(point_ls))))
    anomaly_point = []
    res = pd.DataFrame()
    for point in point_dic.keys():
        print(point)
        point_data = data[data['pointname'] == point][-100:]
        if point_data['aqi'].mean() <=10:
            anomaly_point.append(point)
            continue
        X, Y = data_pre(point_data, predict_map[pollution], future, point_dic)
        # X.extend(train_X)
        # Y.extend(train_Y)
        loaded_model = pickle.load(open('LGBboost_pointaqi_Regressor' + str(future) + '.dat', 'rb'))
        predicted_value = loaded_model.predict(X)
        precision_rate = piture_plot(Y, predicted_value, point)
        res = res.append({'pointname': point, 'precision': precision_rate}, ignore_index=True)
    res.to_csv('LGBboost_pointaqi.csv')


def merge_data():
    import glob
    import os
    import pandas as pd
    inputfile = str(os.path.dirname(os.getcwd()))+"/站点预测/data/广东站点/*.csv"
    outputfile =  str(os.path.dirname(os.getcwd()))+"/站点预测/data/广东站点/merge.csv"
    csv_list = glob.glob(inputfile)
    print(csv_list)
    filepath = csv_list[0]
    df = pd.read_csv(filepath)
    df = df.to_csv(outputfile, index=False)

    for i in range(1, len(csv_list)):
        filepath = csv_list[i]
        df = pd.read_csv(filepath)
        df = df.to_csv(outputfile, index=False, header=False, mode='a+')
if __name__ == '__main__':
#设置固定参数,采样方式1、smote，2、up_sample 3、none

    # train()
    test()

