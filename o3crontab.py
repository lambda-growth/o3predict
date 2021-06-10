# -*- coding:utf-8 -*-
import pickle
from LGBclassifyday import data_pre
import pandas as pd
import json
import requests
import base64
import numpy as np
import datetime
import time
import os
from sklearn.metrics import mean_squared_error

#获取实时的预报未来五天预报结果，生成csv文件

def zq(    start_time, end_time, types = 'DAY', city = None, method = 'CITY'):
   
    url = 'https://www.zq12369.com/api/lnzapi.php'
    params = {
        'appId':'adf4da6927fc8840aec0bd74f04aec61',
        'method':'GET%sDATA' % method,#GETCITYDATA
        'startTime':start_time,
        'endTime':end_time,
        'type':types,
        }
    if city is not None:
        params['city'] = city
    requests.adapters.DEFAULT_RETRIES = 5
    # params = bytes(urlencode(params),encoding='utf8')
    # req = urllib.request.Request(url=url,data=params)
    # data = urlopen(req).read()
    data = requests.post(url=url,data=params).content
    data = base64.b64decode(data)
    data = json.loads(data)
    # print(data)
    data = tdata(data,types,method)
    return data


def tdata(result, types='DAY', method='CITY'):

    result = [pd.Series(row) for row in result]
    result = pd.DataFrame(result)
    result = result.drop_duplicates()
    result = result.fillna(10000)
    if (types == 'DAY') and (method == 'POINT'):
        result['d'] = result['time'].map(lambda g: g[-2:])
        result = result.query('d!="00"')
        result.drop(['d'], axis=1, inplace=True)
    result.set_index('time', inplace=True)

    result.index = pd.to_datetime(result.index)

    if method == 'CITY':
        if types == 'DAY':
            result[['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3_8h', 'temp_avg', 'temp_high', 'temp_low', 'humi',
                    'windangle', 'windlevel',
                    'rain']] = result[
                ['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3_8h', 'temp_avg', 'temp_high', 'temp_low', 'humi',
                 'windangle', 'windlevel',
                 'rain']].applymap(float)
        elif types == 'HOUR':
            result[['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3', 'temp', 'humi', 'windangle', 'windlevel', 'rain',
                    'dswrf']] = result[
                ['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3', 'temp', 'humi', 'windangle', 'windlevel', 'rain',
                 'dswrf']].applymap(float)

        result = result.replace(10000, np.nan)
        return result
    elif method == 'POINT':
        if types == 'DAY':
            result[['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3_8h', 'longitude', 'latitude']] = result[
                ['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3_8h', 'longitude', 'latitude']].applymap(float)
        elif types == 'HOUR':
            result[['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3', 'longitude', 'latitude', 'dswrf']] = result[
                ['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3', 'longitude', 'latitude', 'dswrf']].applymap(float)
        result = result.replace(10000, np.nan)
        return result
def zq_feature(start_time, end_time, city = None):
    url = 'https://www.zq12369.com/api/lnzapi.php'
    params = {
        'appId':'adf4da6927fc8840aec0bd74f04aec61',
        'method':'GETGFSDATA',#GETCITYDATA
        'startTime':start_time,
        'endTime':end_time,
        }
    if city is not None:
        params['city'] = city
    requests.adapters.DEFAULT_RETRIES = 5
    # params = bytes(urlencode(params),encoding='utf8')
    # req = urllib.request.Request(url=url,data=params)
    # data = urlopen(req).read()
    data = requests.post(url=url,data=params).content
    data = base64.b64decode(data)
    result = json.loads(data)
    result = [pd.Series(row) for row in result]
    result = pd.DataFrame(result)
    for fea in ['wda', 'temp','dswrf','humi','apcp']:
        result[fea] = result[fea].astype('int32')
    result = result.drop_duplicates()
    result = result.fillna(10000)
    return result

def dataextract(data):

    data.loc[:,'time'] =  pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    data = data.loc[( data['time']>= '2017-06-21 00:00:00') &
              (data['time'] <= '2020-12-31 23:00:00')].copy()
    data['year'] = data['time'].dt.year
    data['month'] = data['time'].dt.month
    data['day'] = data['time'].dt.day
    data['hour'] = data['time'].dt.hour
    rain_value = data.groupby(['year', 'month', 'day'])['rain'].sum().values
    print(len(rain_value))
    data = data[( data['hour']>= 12) &
              (data['hour'] <= 16)]
    date = pd.date_range('20170621','20201231')
    print(date)
    print(len(date))
    def data_ex(data,fea):
        fea_a = data.groupby(['year','month','day'])[fea].agg('max').values
        fea_b = data.groupby(['year', 'month', 'day'])[fea].agg('mean').values
        fea_c = data.groupby(['year', 'month', 'day'])[fea].agg('min').values
        return fea_a, fea_b, fea_c
    feature_set = {}
    for fea in ['windlevel', 'humi','dswrf']:
        feature_set[fea+'_max'],feature_set[fea+'_mean'],feature_set[fea+'_min'] = data_ex(data, fea)
    dswrf = pd.DataFrame(feature_set)
    dswrf['time'] = date
    dswrf['rain_cum'] = rain_value

    return dswrf

def ensure_float(x):
    import numpy as np
    if np.isnan(x):
        return x
    else :
        return 0


from sklearn.ensemble import RandomForestRegressor


def set_missing_ages(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:, 0]  
    x = known_age[:, 1:]  
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)
    predictedAges = rfr.predict(unknown_age[:, 1:])
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    return df, rfr

def data_ex(data,fea):
    fea_a = data.groupby(['date'])[fea].agg('max').values
    fea_b = data.groupby(['date'])[fea].agg('mean').values
    fea_c = data.groupby(['date'])[fea].agg('min').values
    return fea_a, fea_b, fea_c

def dataextract_fea(data,s_date,e_date,e1_date):
    date = pd.date_range(s_date,e_date, freq='1H')
    date = pd.Series(date,name='time')
    data.loc[:, 'time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    data = pd.merge(date, data,how='left', on = 'time')


    data = data.loc[(data['time'] >= s_date) &
                    (data['time'] < e_date)].copy()
    # data['date'] = data['time']
    # data = data.set_index('date')
    # data = data.set_index(pd.to_datetime(data.index))

    data['year'] = data['time'].dt.year
    data['month'] = data['time'].dt.month
    data['day'] = data['time'].dt.day
    data['hour'] = data['time'].dt.hour
    data['date'] = data['time'].dt.date
    data = data[( data['hour']>= 12) &
              (data['hour'] <= 16)]
    data['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    data.fillna(0,inplace=True)
    # null_all = data.isnull().sum()
    feature_set = {}
    for fea in ['wda', 'temp','dswrf','humi','apcp']:

        feature_set[fea+'_max'],feature_set[fea+'_mean'],feature_set[fea+'_min'] = data_ex(data, fea)
    # data.dropna(inplace=True)
    dswrf = pd.DataFrame(feature_set)

    dswrf['time'] = pd.date_range(s_date,e1_date)
    return dswrf
def city_zq(city=None,future=1,dtime='2021-3-25'):
    now = datetime.datetime.strptime(dtime, "%Y-%m-%d")
    e_date = now + datetime.timedelta(days=future)
    e_date = e_date.strftime("%Y-%m-%d")
    e1_date = now + datetime.timedelta(days=future-1)
    e1_date = e1_date.strftime("%Y-%m-%d")
    s_date= now - datetime.timedelta(days=30)
    s_date = s_date.strftime("%Y-%m-%d")

    aqi_data = zq(s_date,e_date, types='DAY', city=city, method='CITY')
    aqi_data = aqi_data[aqi_data['cityname'].isin(['广州','韶关','深圳','珠海','汕头','佛山','江门','湛江',
                                                    '茂名','肇庆','惠州','梅州','汕尾','阳江','河源','清远',
                                                    '东莞','中山','东沙群岛','潮州','揭阳','云浮'])]
    aqi_data.reset_index(inplace=True)
    expand_data = zq_feature(s_date,e_date,city=city)
    expand_data = dataextract_fea(expand_data,s_date,e_date,e1_date)
    aqi_data.loc[:, 'time'] = pd.to_datetime(aqi_data['time'], format='%Y-%m-%d')
    aqi_data = aqi_data[aqi_data['cityname'] == city].copy()
    city_data = aqi_data.loc[:, ['aqi', 'cityname', 'time', 'co', 'no2', 'so2', 'o3_8h', 'pm2_5', 'pm10']]
    # print(expand_data)
    data = pd.merge(city_data, expand_data, how='right', on='time')
    data.drop_duplicates(subset='time', inplace=True)
    data.loc[:,'cityname'] = city
    # data.to_csv('./2021' + city + '日.csv')
    return data
if __name__ == '__main__':
    citylist = ['广州','韶关','深圳','珠海','汕头','佛山','江门','湛江',
                               '茂名','肇庆','惠州','梅州','汕尾','阳江','河源','清远',
                               '东莞','中山','潮州','揭阳','云浮']
    gen = pd.DataFrame()
    predict_map = {'aqi': 0, 'co': 1, 'no2': 2, 'so2': 3, 'o3_8h': 4, 'pm2_5': 5, 'pm10': 6}
    pollution = 'o3_8h'
    now_str = time.strftime("%Y-%m-%d", time.localtime())
    now = datetime.datetime.strptime(now_str, "%Y-%m-%d")
    end = now + datetime.timedelta(days=5)
    e = {'time': pd.date_range(start=now_str, end=end)}
    city_ls = []
    predict_ls = []
    dtime_ls = []
    for city in citylist:
        for future in [1,2,3,4,5]:
            dtime = now + datetime.timedelta(days=future-1)
            dtime = dtime.strftime('%Y-%m-%d')
            data = city_zq(city,future,dtime)
            # time.sleep(1)
            # data = pd.read_csv('./2021'+city+'日.csv')
            # data = data.iloc[-500:,:]
            X,Y = data_pre(data,predict_map[pollution],future)
            loaded_model = pickle.load(open('./LGBboost_'+pollution+'day_predict'+str(future)+'.dat', 'rb'))
            predicted_value = loaded_model.predict(X)
            e['predict_value' + str(future)] = predicted_value
            print('模型运行预报时间：{}，预报日{}'.format(now_str, dtime))
            print(city+'predict:',predicted_value[-1])
            # print('real',Y[-1])
            dtime_ls.append(dtime)
            predict_ls.append(predicted_value[-1])
            city_ls.append(city)
            precision = 0
            for i in range(len(predicted_value)):
                if Y[i]-15 <= predicted_value[i] <= Y[i]+15:
                    precision += 1

            precision_rate = precision / len(predicted_value)


    a = {'time': dtime_ls, 'o3_8h':predict_ls, 'cityname': city_ls}
    gen = pd.DataFrame(a)
    gen.to_csv('predict_o3_result.csv')



