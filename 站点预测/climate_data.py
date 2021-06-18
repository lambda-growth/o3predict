import datetime
import math

import pandas as pd
# 提取小时数据中的紫外等特征转成天数据，合并到六项中
import json
import requests
import base64
import numpy as np
import base

def zq_feature(start_time, end_time, city = None):
    """
    获取空气质量与气象数据并做初步处理
    Params:
    ----------
    start_time: 开始日期
    end_time: 结束日期
    city: 城市列表



    Return:
    ----------
    汇总后的数据列表
    """
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
def data_ex(data,fea):
    fea_a = data.groupby(['date'])[fea].agg('max').values
    fea_b = data.groupby(['date'])[fea].agg('mean').values
    fea_c = data.groupby(['date'])[fea].agg('min').values
    return fea_a, fea_b, fea_c

def dataextract_fea(data, s_time, e2_time, e_time,city):
    date = pd.date_range(s_time, e2_time, freq='1H')
    date = pd.Series(date,name='time')
    data.loc[:, 'time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    data = pd.merge(date, data,how='left', on = 'time')
    # data = data.loc[(data['time'] >= '2021-01-01 00:00:00') &
    #                 (data['time'] <= '2021-03-10 23:00:00')].copy()
    data = data.loc[(data['time'] >= s_time) &
                    (data['time'] < e2_time)].copy()
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

    # data['wu'] = np.cos((270 - data['wda']) * math.pi / 180) * data['ws'].astype('float')
    # data['wv'] = np.sin((270 - data['wda']) * math.pi / 180) * data['ws'].astype('float')
    feature_set = {}
    for fea in ['wda','temp','dswrf','humi','apcp']:
        # 插值法填充
        # data[fea] = data[fea].fillna(data[fea].mean()) #均值填充
        # data[fea] = data[fea].interpolate()
        feature_set[fea+'_max'],feature_set[fea+'_mean'],feature_set[fea+'_min'] = data_ex(data, fea)
    # data.dropna(inplace=True)
    dswrf = pd.DataFrame(feature_set)
    dswrf['time'] = pd.date_range(s_time, e_time)
    dswrf['cityname'] = city
    return dswrf

citylist = base.area_group('广东')
s_time = '2017-10-31'
e_time = '2021-6-15' #返回的数据是【2017.10.31--2021.6.14】日的
e2_time = datetime.datetime.strptime(e_time, "%Y-%m-%d")
e2_time += datetime.timedelta(days=1)
e2_time.strftime("%Y-%m-%d")

for city in citylist:
    data_fea = zq_feature(s_time, e_time, city=city)
    # 小时数据特征提取，转化日数据
    expand_data = dataextract_fea(data_fea, s_time, e2_time, e_time, city)
    # 将污染日数据和未来气象数据合并、保存
    data = pd.read_csv('./data/广东站点/merge.csv')
    data.loc[:, 'time'] = pd.to_datetime(data['time'], format='%Y-%m-%d')
    city_data = data[data['cityname'] == city].copy()

    data = pd.merge(city_data, expand_data, how='left', on=['time','cityname'])
    data.to_csv('./data/广东站点气象/' + city + '站点日.csv')
