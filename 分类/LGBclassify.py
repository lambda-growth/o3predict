import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, ShuffleSplit
from datetime import datetime
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_squared_error, classification_report, f1_score, make_scorer, accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#1. 定义特征工程函数
def data_pre(df,pollution,future,type=None):
    # 滑动串口统计特征
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
    #数据筛选
    data = df.loc[:, ['aqi','co','no2','so2','o3_8h','pm2_5','pm10','cityname','time','wda_max',
                      'wda_mean', 'wda_min', 'temp_max', 'temp_mean', 'temp_min', 'dswrf_max', 'dswrf_mean', 'dswrf_min', 'humi_max', 'humi_mean', 'humi_min']]
    # 城市类别处理（类似one_hot）
    data['cityname'] = data['cityname'].map({'广州':0,'韶关':1,'深圳':2,'汕头':3,'珠海':4,'佛山':5,'江门':6,'湛江':7,
                '茂名':8,'肇庆':9,'惠州':10,'梅州':11,'汕尾':12,'阳江':13,'河源':14,'清远':15,
                '东莞':16,'中山':17,'潮州':18,'揭阳':19,'云浮':20}).astype(int)
    # 原始时间数据转datatime，提取月特征
    data.loc[:,'time'] =  pd.to_datetime(data['time'], format='%Y-%m-%d')
    data['month'] = data['time'].dt.month
    # 假期
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
    # second: one hot encode to 7 columns
    data = data.join(pd.get_dummies(to_one_hot))
    # data = data[(data['month']==6)|(data['month']==7)|(data['month']==8)]
    # data = data.join(pd.get_dummies(data[['winddirect','weather']])) #日数据中的类别特征，未来数据不含，删除
    # data.drop(columns= ['winddirect','weather'], inplace=True)
#获取特征的历史平均，最大最小值,'dswrf_max','humi_max','apcp_max'
    for i in [3, 7, 14]:
        for j in ['wda_max','temp_max','dswrf_max','humi_max']:
            data = get_mov_avg_std(data,j,i)
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
#获得时间特征
    data.loc[:,'time'] =  pd.to_datetime(data['time'], format='%Y-%m-%d')
    data['year'] = data['time'].dt.year
    data['month'] = data['time'].dt.month
    data.drop(['time'], axis=1,inplace=True)
    # data['o3_month_mean'] = data.groupby('month')['o3_8h'].transform('mean')
    # data['o3_month_max'] = data.groupby('month')['o3_8h'].transform('max')

#diff()求差分
    # data[]
#臭氧历史特征，特征偏移
    if future != 0:

        for k in [future,future+1,future+2]:
            data['aqi_shift'+str(k)] = data['aqi'].shift(k)
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
#空值填充，删除无前值行

    data.fillna(method='ffill',inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.dropna(axis=0, inplace=True)




# 获取类别标签
    def func(x):
        if x <= 160:
            return 0
        else:
            return 1

    data[pollution] = data[pollution].map(func)


 # 采样方法
    def up_sample(df):
        df1 = df[df[pollution] == 1]  # 正例
        df2 = df[df[pollution] == 0]  ##负例
        df3 = pd.concat([df1, df1, df1, df1, df1], ignore_index=True)
        return pd.concat([df2, df3], ignore_index=True)

    def down_sample(df):
        df1 = df[df[pollution] == 1]  # 正例
        df2 = df[df[pollution] == 0]  ##负例
        df3 = df2.sample(frac=0.25)  ##抽负例
        return pd.concat([df1, df3], ignore_index=True)
    if type == 'up_sample':
        data = up_sample(data)
        columns = data.columns
    # print(data)
    # data1 = data.copy()
    # columns = ['aqi', 'co', 'no2', 'so2', 'o3_8h', 'pm2_5', 'pm10', 'cityname',
    #        'wda_max', 'wda_mean',
    #        ...]
    # for column in columns:
    #     data = data1.drop([column],axis=1)
    predict_map = {'aqi': 0, 'co': 1, 'no2': 2, 'so2': 3, 'o3_8h': 4, 'pm2_5': 5, 'pm10': 6}
    data = np.array(data)
    X, Y = data[:,7:].tolist(),data[:,predict_map[pollution]].tolist()
    return X,Y
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
def train_xgb():
    import xgboost as xgb
    X, Y = [], []

    pollution = 'o3_8h'
    future = 0
    citys = ['广州', '韶关', '深圳', '珠海', '佛山', '江门', '湛江',
             '茂名', '肇庆', '惠州', '梅州', '汕尾', '汕头', '阳江', '河源', '清远',
             '东莞', '中山', '潮州', '揭阳', '云浮']

    for city in citys:
        data = pd.read_csv('../data/2020' + city + '日.csv').iloc[:-128, :]
        # data.drop(index=(data.loc[(data['humi_max']==0)].index))
        # data.drop(index=(data.loc[(data['dswrf_max'] == 0)].index))

        # data = pd.read_csv('../CityData/datalab/广东/2020'+city+'日.csv')
        # data = data.iloc[:-500,:]
        # train_X, train_Y = data_pre(data,type='up_sample')
        train_X, train_Y = data_pre(data,pollution, future)
        # train_X, train_Y = data_pre(data, predict_map[pollution], future, type='up_sample')
        X.extend(train_X)
        Y.extend(train_Y)

    # data = pd.read_csv('../CityData/datalab/广东/2020' + city + '日.csv')
    # data = pd.read_csv('../CityData/datalab/广东/2020' + '广州' + '日.csv')
    # X, Y = data_pre(data)
    X = np.array(X)
    Y = np.array(Y)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, y_test)


    param = {'max_depth': 5, 'eta': 0.5, 'verbosity': 1, 'objective': 'binary:logistic','scale_pos_weight':100}
    xgb_model_ = xgb.XGBClassifier(**param)
    xgb_model_.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="error",
                   eval_set=[(X_test, y_test)])
    predicted_value = xgb_model_.predict(X_test)
    print(classification_report(
        y_test, predicted_value, labels=[0, 1], target_names=["优良", "轻度污染"]))

    pickle.dump(xgb_model_, open('xgboost_aqi_classify' + str(future) + '.dat', 'wb'))

def train():
    X, Y = [], []
    predict_map = {'aqi': 0, 'co': 1, 'no2': 2, 'so2': 3, 'o3_8h': 4, 'pm2_5': 5, 'pm10': 6}
    pollution = 'o3_8h'
    future = 0
    citys = ['广州', '韶关', '深圳', '珠海', '佛山', '江门', '湛江',
             '茂名', '肇庆', '惠州', '梅州', '汕尾', '汕头', '阳江', '河源', '清远',
             '东莞', '中山', '潮州', '揭阳', '云浮']

    for city in citys:
        data = pd.read_csv('./data/2020' + city + '日.csv').iloc[:-128, :]
        # data.drop(index=(data.loc[(data['humi_max']==0)].index))
        # data.drop(index=(data.loc[(data['dswrf_max'] == 0)].index))

        # data = pd.read_csv('../CityData/datalab/广东/2020'+city+'日.csv')
        # data = data.iloc[:-500,:]
        # train_X, train_Y = data_pre(data,type='up_sample')
        # train_X, train_Y = data_pre(data, predict_map[pollution], future)
        train_X, train_Y = data_pre(data, predict_map[pollution], future, type='up_sample')
        X.extend(train_X)
        Y.extend(train_Y)

    # data = pd.read_csv('../CityData/datalab/广东/2020' + city + '日.csv')
    # data = pd.read_csv('../CityData/datalab/广东/2020' + '广州' + '日.csv')
    # X, Y = data_pre(data)
    X = np.array(X)
    Y = np.array(Y)
    print(Y)
    from collections import Counter

    weights = {'0': 1 - (np.array(Y).sum() / len(Y)), '1': np.array(Y).sum() / len(Y)}

    from collections import Counter

    print(Counter(Y))
    # 使用imlbearn库中上采样方法中的SMOTE接口
    from imblearn.over_sampling import SMOTE

    # 定义SMOTE模型，random_state相当于随机数种子的作用
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_sample(X, Y)
    print(Counter(y_smo))
    print(X_smo, y_smo)
    # X_train, X_vaild, y_train, y_vaild = train_test_split(X_smo, y_smo, test_size=0.3, random_state=42)
    X_train, X_vaild, y_train, y_vaild = train_test_split(X, Y, test_size=0.3, random_state=42)

    # LGBM
    #     lgbm_clf = LGBMClassifier(
    #         # **params
    #         random_state=626,
    #         n_estimators=800,
    #         learning_rate=0.1,
    #         max_depth=-1,
    #         num_leaves=127,
    #         colsample_bytree=0.8,
    #         subsample=0.8,
    #         lambda_l1=0.1,  # 0.1
    #         lambda_l2=0.2,  # 0.2
    #         # gpu_platform_id=1,
    #     )
    #     from bayes_opt import BayesianOptimization
    #
    #
    #     # 定义优化参数
    #     def rf_cv(max_depth, subsample, num_leaves, colsample_bytree):
    #         val = cross_val_score(LGBMClassifier(
    #             n_estimators=300,
    #             # learning_rate=0.1,
    #             max_depth=int(max_depth),
    #             subsample=min(subsample, 0.999),
    #             num_leaves=int(num_leaves),
    #             colsample_bytree=min(colsample_bytree, 0.99),
    #             # lambda_l1= 0.1,   # 0.1
    #             # lambda_l2=0.2,  # 0.2
    #             random_state=626,
    #             # device='gpu:01',
    #             # gpu_platform_id=1 ),
    #             X_train, y_train, scoring="accuracy", cv=5).mean()
    #         return val
    #
    #
    # # 贝叶斯优化
    #     rf_bo = BayesianOptimization(rf_cv,
    #                                  {
    #                                       # "n_estimators":(100,300),
    #                                       # 'learning_rate':(0.08,0.2),
    #                                      "colsample_bytree": (0.7, 0.9),
    #                                      "subsample": (0.7, 0.9),
    #                                      "max_depth": (5, 11),
    #                                      'num_leaves': (31, 150),
    #                                      # 'lambda_l1':(0.1,0.3),
    #                                      # 'lambda_l2':(0.1,0.3),
    #                                  })
    #     # 开始优化
    #
    #
    #     num_iter = 25
    #     init_points = 4
    #     rf_bo.maximize(init_points=init_points, n_iter=num_iter)
    #     # 显示优化结果
    #     print(rf_bo.max)
    def BalancedSampleWeights(y_train):
        classes = np.unique(y_train, axis=0)

        classes.sort()

        class_samples = np.bincount(y_train)

        total_samples = class_samples.sum()

        n_classes = len(class_samples)

        weights = total_samples / (n_classes * class_samples * 1.0)
        weights = dict(zip([0, 1], weights))

        return weights  # Usage

    weight = BalancedSampleWeights(np.array(Y, dtype=int))

    lgb_clf = LGBMClassifier(
        n_jobs=-1,
        # device_type='gpu',
        n_estimators=400,
        learning_rate=0.1,
        max_depth=11,
        num_leaves=109,
        colsample_bytree=0.8180764153843182,
        subsample=0.729,
        class_weight='balanced'
        # class_weight= weight
        # max_bins=127,
    )

    scores = cross_val_score(lgb_clf, X=X_train, y=y_train, verbose=1, cv=5, scoring=make_scorer(accuracy_score),
                             n_jobs=-1)
    scores.mean()
    # 神经网络
    # predicted_value, load_model = create_model(X_train, y_train, X_test)

    lgb_clf.fit(X_train, y_train)
    predicted_value = lgb_clf.predict(X_vaild)
    print("lgb_clf：", np.mean(predicted_value == y_vaild))
    print(classification_report(
        y_vaild, predicted_value, labels=[0, 1], target_names=["优良", "轻度污染"]))
    # piture_plot(y_test, predicted_value)
    # final_predicted_value.append(predicted_value)
    # piture_plot(y_test, np.array(final_predicted_value).mean(axis=0))
    # 0.7747, 0.720377,0.720894, 0.730622,0.728034
    # 保存模型参数
    # pickle.dump(model, open('LGBclasify_aqiday_predict' + str(future) + '.dat', 'wb'))
    pickle.dump(lgb_clf, open('LGBboost_aqi_classify' + str(future) + '.dat', 'wb'))
    # load_model.save_weights("o3day_model.h5")

def test():
    X, Y = [], []
    predict_map = {'aqi': 0, 'co': 1, 'no2': 2, 'so2': 3, 'o3_8h': 4, 'pm2_5': 5, 'pm10': 6}
    pollution = 'o3_8h'
    future = 0
    citys = ['广州', '韶关', '深圳', '珠海', '佛山', '江门', '湛江',
             '茂名', '肇庆', '惠州', '梅州', '汕尾', '汕头', '阳江', '河源', '清远',
             '东莞', '中山', '潮州', '揭阳', '云浮']

    for city in citys:
        data = pd.read_csv('../data/2020' + city + '日.csv').iloc[-128:, :]
        test_X, test_Y = data_pre(data, pollution, future)

        X.extend(test_X)
        Y.extend(test_Y)

    # data = pd.read_csv('../CityData/datalab/广东/2020' + city + '日.csv')
    # data = pd.read_csv('../CityData/datalab/广东/2020' + '广州' + '日.csv')
    # X, Y = data_pre(data)
    X = np.array(X)
    Y = np.array(Y)
    loaded_model = pickle.load(open('xgboost_aqi_classify' + str(future) + '.dat', 'rb'))
    result = loaded_model.predict(X)
    print(classification_report(
        Y, result, labels=[0, 1], target_names=["优良", "轻度污染"]))
if __name__ == '__main__':
#设置固定参数,采样方式1、smote，2、up_sample 3、none
    train_xgb()
    # train()
    test()
    # X, Y = [], []
    # predict_map = {'aqi': 0, 'co': 1, 'no2': 2, 'so2': 3, 'o3_8h': 4, 'pm2_5': 5, 'pm10': 6}
    # pollution = 'aqi'
    # future = 0
    # loaded_model = pickle.load(open('xgboost_aqi_classify' + str(future) + '.dat', 'rb'))
    # citys = ['广州', '韶关', '深圳', '珠海', '佛山', '江门', '湛江',
    #          '茂名', '肇庆', '惠州', '梅州', '汕尾', '汕头', '阳江', '河源', '清远',
    #          '东莞', '中山', '潮州', '揭阳', '云浮']
    #
    # for city in ['汕头']:
    #     data = pd.read_csv('./data/2020' + city + '日.csv').iloc[-493:, :]
    #     test_X, test_Y = data_pre(data, predict_map[pollution], future)
    #
    #     X = np.array(test_X)
    #     Y = np.array(test_Y)
    #
    #     result = loaded_model.predict(X)
    #     print(city)
    #     print(classification_report(
    #         Y, result, labels=[0, 1], target_names=["优良", "轻度污染"]))
