import argparse
import pickle
import joblib
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
import base
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#1. load dataset


def data_pre(df,pollution,future):

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
        df_out[col + str(N) + '_mean'] = mean_list
        df_out[col + str(N) + '_skew'] = skew_list
        df_out[col + str(N) + '_median'] = median_list
        df_out[col + str(N) + '_std'] = std_list
        df_out[col + str(N) + '_max'] = max_list
        df_out[col + str(N) + '_min'] = min_list
        # df_out[col + str(N)+'_q10'] = q10_list
        # df_out[col + str(N) + '_q50'] = q50_list
        # df_out[col + str(N) + '_q90'] = q90_list
        return df_out

    # 新增小时统计特征：'windlevel_max', 'windlevel_mean','humi_max','humi_mean', 'dswrf_max', 'dswrf_mean'
    # data = df.loc[:,['aqi','cityname','time','temp_avg','windangle','windlevel','rain','humi','windlevel_max', 'windlevel_mean','humi_max',
    #    'humi_mean', 'dswrf_max', 'dswrf_mean' ]]
    data = df.loc[:,
           ['aqi', 'co', 'no2', 'so2', 'o3_8h', 'pm2_5', 'pm10', 'cityname', 'time', 'wda_max', 'wda_mean',
            'wda_min', 'temp_max', 'temp_mean', 'temp_min', 'dswrf_max', 'dswrf_mean', 'dswrf_min', 'humi_max',
            'humi_mean', 'humi_min']]
    # 原特征
    # data = df.loc[:,['o3_8h', 'cityname','time', 'temp_avg', 'windangle', 'windlevel', 'rain', 'humi','dswrf_max']]
    # data['citylocate'] = data['cityname'].map({'广州':1,'韶关':2,'深圳':1,'汕头':3,'珠海':1,'佛山':1,'江门':1,'湛江':4,
    #             '茂名':4,'肇庆':1,'惠州':1,'梅州':2,'汕尾':3,'阳江':4,'河源':2,'清远':2,
    #             '东莞':1,'中山':1,'潮州':3,'揭阳':3,'云浮':2})
    data['cityname'] = data['cityname'].map({'广州': 0, '韶关': 1, '深圳': 2, '汕头': 3, '珠海': 4, '佛山': 5, '江门': 6, '湛江': 7,
                                             '茂名': 8, '肇庆': 9, '惠州': 10, '梅州': 11, '汕尾': 12, '阳江': 13, '河源': 14,
                                             '清远': 15,
                                             '东莞': 16, '中山': 17, '潮州': 18, '揭阳': 19, '云浮': 20}).astype(int)

    data.loc[:, 'time'] = pd.to_datetime(data['time'], format='%Y-%m-%d')
    data['month'] = data['time'].dt.month
    # 筛选月份

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
                                                                    '2019-01-01', '2019-02-04', '2019-02-05',
                          '2019-02-06', '2019-02-07',
                          '2019-02-08', '2019-02-09', '2019-02-10', '2019-04-05', '2019-04-06',
                          '2019-04-07', '2019-04-29', '2019-04-30', '2019-05-01', '2019-06-07',
                          '2019-06-08', '2019-06-09', '2019-09-13', '2019-09-14', '2019-09-15',
                          '2019-10-01', '2019-10-02', '2019-10-03', '2019-10-04', '2019-10-05',
                          '2019-10-06', '2019-10-07', '2019-12-30', '2019-12-31'
                                                                    '2020-01-01', '2020-01-22', '2020-01-23',
                          '2020-01-24', '2020-01-27',
                          '2020-01-28', '2020-01-29', '2020-01-30', '2020-01-31', '2020-02-01',
                          '2020-04-06', '2020-05-01', '2020-05-04', '2020-05-05', '2020-06-25',
                          '2020-06-26', '2020-10-01', '2020-10-02', '2020-10-05', '2020-10-06',
                          '2020-10-07', '2020-10-08', '2021-01-01', '2021-02-11', '2021-02-12',
                          '2021-02-15', '2021-02-16', '2021-02-17', '2021-04-05', '2021-05-03',
                          '2021-05-04', '2021-05-05', '2021-06-14', '2021-09-20', '2021-09-21',
                          '2021-10-01', '2021-10-04', '2021-10-05', '2021-10-06', '2021-10-07',
                          ]
    workdays_exception = [
        '2017-01-22', '2017-02-04', '2017-04-01', '2017-05-27', '2017-09-30',
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

    data['weekday'] = pd.to_datetime(data['time']).apply(lambda x: x.weekday() + 1)
    data['isholiday'] = data['time'].apply(lambda x: int(is_holiday(x)))
    to_one_hot = data['time'].dt.day_name()
    # second: one hot encode to 7 columns
    data = data.join(pd.get_dummies(to_one_hot))
    # data = data[(data['month']==6)|(data['month']==7)|(data['month']==8)]
    # data = data.join(pd.get_dummies(data[['winddirect','weather']])) #日数据中的类别特征，未来数据不含，删除
    # data.drop(columns= ['winddirect','weather'], inplace=True)
    # 获取特征的历史平均，最大最小值,'dswrf_max','humi_max','apcp_max'
    for i in [3, 7, 14]:
        for j in ['wda_max', 'temp_max', 'dswrf_max', 'humi_max']:
            data = get_mov_avg_std(data, j, i)
    # data = get_mov_avg_std(data, 'o3_8h', 30)
    # data = get_mov_avg_std(data, 'temp_avg', 30)
    # data = get_mov_avg_std(data, 'humi', 30)
    # data = get_mov_avg_std(data, 'no2', 30)
    # data = get_mov_avg_std(data, 'no2', 7)
    # data = get_mov_avg_std(data, 'o3_8h', 7)
    # data = get_mov_avg_std(data, 'temp_avg', 7)
    # data = get_mov_avg_std(data, 'humi', 7)
    # data = get_mov_avg_std(data, 'o3_8h', 3)
    # data = get_mov_avg_std(data, 'temp_avg', 3)
    # data = get_mov_avg_std(data, 'humi', 3)
    # data = get_mov_avg_std(data, 'co', 3)
    # data = get_mov_avg_std(data, 'no2', 3)
    # data = get_mov_avg_std(data, 'so2', 3)

    # 统计的指数平滑值
    # from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
    # y_hat_avg = data['o3_8h'].copy()
    # fit2 = SimpleExpSmoothing(np.asarray(data['o3_8h'])).fit(
    # smoothing_level=0.6,optimized=False)
    # # data['SES'] = fit2.forecast(len(data['o3_8h']))

    # print(data.columns)
    # 获得时间特征
    data.loc[:, 'time'] = pd.to_datetime(data['time'], format='%Y-%m-%d')
    data['year'] = data['time'].dt.year
    data['month'] = data['time'].dt.month
    data.drop(['time'], axis=1, inplace=True)

    # data['o3_month_mean'] = data.groupby('month')['o3_8h'].transform('mean')
    # data['o3_month_max'] = data.groupby('month')['o3_8h'].transform('max')
    # 获取类别标签
    def func(x):
        if x <= 160:
            return 0
        else:
            return 1

    # diff()求差分
    # data[]
    # 臭氧历史特征，特征偏移
    for k in [future, future + 1, future + 2]:
        data['aqi_shift' + str(k)] = data['aqi'].shift(k)
        data['o3_shift' + str(k)] = data['o3_8h'].shift(k)
        data['dswrf_max' + str(k)] = data['dswrf_max'].shift(k)
        # data['humi_max' + str(k)] = data['humi_max'].shift(k)
        # data['so2_shift' + str(k)] = data['so2'].shift(k)
        # data['co_shift' + str(k)] = data['co'].shift(k)
        # data['no2_shift' + str(k)] = data['no2'].shift(k)
        # data['pm2_5_shift' + str(k)] = data['pm2_5'].shift(k)
        # data['pm10_shift' + str(k)] = data['pm10'].shift(k)

        # data['humi_shift'+'k'] = data['humi'].shift(k)
        # data['temp_shift'+'k'] = data['temp_high'].shift(k)
        # data['o3_shift4'] = data['o3_8h'].shift(4)
        # data['o3_shift5'] = data['o3_8h'].shift(5)
    # data.dropna(axis=0,inplace=True)
    # 空值填充，删除无前值行

    data.fillna(method='ffill', inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.dropna(axis=0, inplace=True)
    # 采样方法
    columns = data.columns
    # data1 = data.copy()
    # columns = ['aqi', 'co', 'no2', 'so2', 'o3_8h', 'pm2_5', 'pm10', 'cityname',
    #        'wda_max', 'wda_mean',
    # for column in columns:
    #     data = data1.drop([column],axis=1)
    data = np.array(data).astype('float32')
    # np.save('GCN_predict-Pytorch-main/GCN_predict-Pytorch-main/PeMS_04/henanp.npy', data)
    X, Y = data[:,7:].tolist(),data[:,pollution].tolist()
    return X,Y
def piture_plot(y_test, predicted_value):

    # print(len(y_test[60:]))
    y1 = list(np.array(y_test[60:]) - 15)
    y2 = list(np.array(y_test[60:]) + 15)
    plt.rcParams['figure.figsize'] = (12.0, 5.0)
    fig = plt.figure()
    # 画柱形图
    ax1 = fig.add_subplot(111)
    ax1.fill_between(range(len(y_test[60:])), y1, y2, alpha=.08, color='b')
    ax1.plot(y_test[60:], color='red', label='Real aqi value')
    ax1.plot(predicted_value[60:], color='blue', label='Predicted aqi value')
    ax1.set_ylabel('杭州' + 'AQI值', fontsize='15')
    plt.title('future  days aqi Prediction')
    plt.xlabel('Time')
    plt.ylabel('aqi value')
    plt.legend()
    plt.savefig('aqi.png')
    # plt.show()


    wrong_ls = []
    precision = 0
    for i in range(len(predicted_value)):
        if y_test[i] - 15 <= predicted_value[i] <= y_test[i] + 15:
            precision += 1
        else:
            wrong_ls.append(y_test[i])
    sns.distplot(np.array(wrong_ls), kde=False,label = 'false aqi distribution')
    sns.distplot(np.array(y_test),kde=False,label = 'original aqi distribution')
    plt.title('aqi distribution')
    plt.xlabel('aqi value')
    plt.ylabel('number')
    plt.legend()
    plt.savefig('distribution.png')
    # plt.show()
    print(pd.Series(wrong_ls).std())
    precision_rate = precision / len(predicted_value)

    print(mean_squared_error(predicted_value, y_test))
    print('准确率: %.6f' % precision_rate)
    return precision_rate
    # precision_ls = [0.7747, 0.720377,0.720894, 0.730622,0.728034]
    # plt.plot(precision_ls)
    # plt.xticks([0,1,2,3,4],[r'第一天', r'第二天', r'第三天', r'第四天', r'第五天'])
    # plt.title('AQI预测准确率')
    # plt.xlabel('未来天')
    # plt.ylabel('准确率')
    # plt.savefig('precision.png')
    # plt.show()

#0.7747, 0.720377,0.720894, 0.730622,0.728034
#保存模型参数
    # pickle.dump(model,open('LGBboost_o3day_predict.dat','wb'))
class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], self.n_splits))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                print ("Fit Model %d fold %d" % (i, j))
                clf.fit(X_train, y_train)
                # bad_indices = np.where(np.isinf(X))
                # print(bad_indices)
                # print(X[bad_indices])
                y_pred = clf.predict(X_holdout)[:]

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)

        # results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        # print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
        return res

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
def mlp():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((96,)),
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
        model.fit(x_train_, y_train_,epochs=100,batch_size=128,validation_data=(x_valid_,y_valid_ ), verbose=1,
            callbacks=[early_stopping])



        # 模型预测
        valid_pred = model.predict(x_valid_)
        print('MSE: ', mean_squared_error(y_valid_, valid_pred))
        predicted_value += np.array(tf.squeeze(model.predict(X_test)))
    predicted_value /= n_fold
    return predicted_value , model

def mymodel_lgb(X_train,y_train,X_test):
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
    n_fold = 5
    predicted_value = np.zeros(len(X_test))
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train):
        x_train_, x_valid_= X_train[train_idx], X_train[val_idx]
        y_train_, y_valid_ = y_train[train_idx], y_train[val_idx]
        model.fit(x_train_, y_train_)
#         # 模型预测
        valid_pred = model.predict(x_valid_)
        print('MSE: ', mean_squared_error(y_valid_, valid_pred))
        predicted_value += model.predict(X_test)
    predicted_value /= n_fold
    print(predicted_value.mean())
    return predicted_value, model

def mymodel_ensemble(x_train, y_train, x_test):
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    # rf params
    rf_params = {}
    rf_params['n_estimators'] = 50
    rf_params['max_depth'] = 8
    rf_params['min_samples_split'] = 100
    rf_params['min_samples_leaf'] = 30

    # xgb params
    xgb_params = {}
    #xgb_params['n_estimators'] = 50
    xgb_params['min_child_weight'] = 12
    xgb_params['learning_rate'] = 0.37
    xgb_params['max_depth'] = 6
    xgb_params['subsample'] = 0.77
    xgb_params['reg_lambda'] = 0.8
    xgb_params['reg_alpha'] = 0.4
    xgb_params['base_score'] = 0
    #xgb_params['seed'] = 400
    xgb_params['silent'] = 1

    # lgb params
    lgb_params = {}
    #lgb_params['n_estimators'] = 50
    lgb_params['max_bin'] = 8
    lgb_params['learning_rate'] = 0.05 # shrinkage_rate
    lgb_params['metric'] = 'l1'          # or 'mae'
    lgb_params['sub_feature'] = 0.35
    lgb_params['bagging_fraction'] = 0.85 # sub_row
    lgb_params['bagging_freq'] = 40
    lgb_params['num_leaves'] = 512        # num_leaf
    lgb_params['min_data'] = 100         # min_data_in_leaf
    lgb_params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
    lgb_params['verbose'] = 0
    lgb_params['feature_fraction_seed'] = 2
    lgb_params['bagging_seed'] = 3



    # XGB model
    xgb_model = XGBRegressor(**xgb_params)

    # lgb model
    lgb_model = LGBMRegressor(**lgb_params)

    # RF model
    rf_model = RandomForestRegressor(**rf_params)

    # ET model
    et_model = ExtraTreesRegressor()

    # SVR model
    # SVM is too slow in more then 10000 set
    #svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.05)

    # DecsionTree model
    dt_model = DecisionTreeRegressor()

    # AdaBoost model
    ada_model = AdaBoostRegressor()
    stack = StackingAveragedModels(base_models=(rf_model, xgb_model),
                           meta_model=LinearRegression(),n_folds=3)
    # stack= Ensemble(n_splits=5,
    #         stacker=LinearRegression(),
    #         base_models=(rf_model, xgb_model, lgb_model, et_model, ada_model))
    # y_test = stack.fit_predict(x_train, y_train, x_test)
    stack.fit(x_train, y_train)
    joblib.dump(stack, './stack.pkl')
    y_test = stack.predict(x_test)
    return y_test,stack


from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
import numpy as np
#对于分类问题可以使用 ClassifierMixin


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=3):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # 我们将原来的模型clone出来，并且进行实现fit功能
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        folds = list(KFold(n_splits=self.n_folds, shuffle=True, random_state=2016).split(X, y))

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                print("Fit Model %d fold %d" % (i, j))
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]

                out_of_fold_predictions[test_idx, i] = y_pred
        # 使用次级训练集来训练次级学习器
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    #在上面的fit方法当中，我们已经将我们训练出来的初级学习器和次级学习器保存下来了
    #predict的时候只需要用这些学习器构造我们的次级预测数据集并且进行预测就可以了
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

def f1(x):
    return np.log(x+1)
def rf1(x):
    return np.exp(x)-1

def train():
    predict_map = {'aqi':0, 'co':1, 'no2':2, 'so2':3, 'o3_8h':4, 'pm2_5':5, 'pm10':6}
    pollution = 'aqi'
    future = 5
    X,Y = [],[]
    area = '广东'
    # data = pd.read_csv('河南all1.csv').sort_values(by=['cityname', 'time'])
    # X, Y = data_pre(data,predict_map[pollution])
    citylist = base.area_group(area)
    for city in citylist:
        data = pd.read_csv('./data/2020' + city + '日.csv').iloc[:-128,:]
        train_X, train_Y = data_pre(data,predict_map[pollution],future)

        X.extend(train_X)
        Y.extend(train_Y)

    # data = np.load('./GCN_predict-Pytorch-main/GCN_predict-Pytorch-main/PeMS_04/henanp.npy')
    # X = data[:,:,7:].reshape(21369,88)
    # Y = data[:,:,:1].reshape(21369,)
    # print(data.shape)
    # data = pd.read_csv('../CityData/datalab/广东/2020' + city + '日.csv')
    # data = pd.read_csv('../CityData/datalab/广东/2020' + '广州' + '日.csv')
    # X, Y = data_pre(data)
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)
    X[np.isinf(X)] = 0
    from collections import Counter
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    predicted_value,load_model  = mymodel_lgb(X_train,y_train,X_test)
    # predicted_value, load_model = mymodel_ensemble(X_train, y_train, X_test)
    # predicted_value,load_model = create_model(X_train,y_train,X_test)
    pickle.dump(load_model, open('LGBboost_'+pollution+'day_predict' + str(future) + '.dat', 'wb'))
    piture_plot(y_test, predicted_value)

    return load_model
# def predict(model , city,future):
#     area = '广东'
#     predict_map = {'aqi': 0, 'co': 1, 'no2': 2, 'so2': 3, 'o3_8h': 4, 'pm2_5': 5, 'pm10': 6}
#
#     # pollution 对应map
#     pollution = 'o3_8h'
#     future = 5
#     data = pd.read_csv('./data/2020' + city + '日.csv').iloc[-128:,:]
#     # data = pd.read_csv('../CityData/datalab/' + area + '/2021' + city + '2017-6-23--2021-4-20日.csv').iloc[-100:,:]
#     loaded_model = model
#     X, Y = data_pre(data, predict_map[pollution], future)
#     predicted_value = loaded_model.predict(X)
#     precision_rate = piture_plot(Y,predicted_value)
#     return precision_rate

if __name__ == '__main__':
#设置固定参数,采样方式1、smote，2、up_sample 3、none
    train()


