import pickle
import xgboost as xgb
import requests
from LGBregression import data_pre
from LGBregression import mlp
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt


precision_ls = []
# for city in ['广州', '韶关', '深圳', '珠海', '汕头', '佛山', '江门', '湛江',
#              '茂名', '肇庆', '惠州', '梅州', '汕尾', '阳江', '河源', '清远',
#              '东莞', '中山', '潮州', '揭阳', '云浮']:
for city in ['汕头']:
    e = {'time': pd.date_range(start='20210101', end='20210507')}
    for future in [1,2,3,4,5,6,7]:
        predict_map = {'aqi':0, 'co':1, 'no2':2, 'so2':3, 'o3_8h':4, 'pm2_5':5, 'pm10':6}
        pollution = 'aqi'
        data = pd.read_csv('../data/2020'+city+'日.csv').iloc[-127-14:,:]
        # data = data.iloc[-500:,:]
        X,Y = data_pre(data,predict_map[pollution],future)




        loaded_model = pickle.load(open('model/LGBboost_'+pollution+'day_predict'+str(future)+'.dat', 'rb'))
        predicted_value = loaded_model.predict(X)[-127:]
        Y = Y[-127:]


        print(predicted_value[-10:])
        wrong_ls = []
        precision = 0
        for i in range(len(predicted_value)):
            if Y[i]-15 <= predicted_value[i] <= Y[i]+15:
                precision += 1
            else:
                wrong_ls.append(Y[i])
        e['predict_value'+str(future)] = predicted_value
        precision_rate = precision / len(predicted_value)
        precision_ls.append(precision_rate)

        print(mean_squared_error(predicted_value, Y))
        print('城市{}准确率: {:.3f}'.format(city,precision_rate))
    e['real'] = Y
    df = pd.DataFrame(e)
    # df.to_csv('../图/'+pollution+'预测/'+city+'predict_'+str(future)+pollution+'value_day.csv',index=None)

    # df.to_csv('../图/'+pollution+'预测/'+city+'predict_'+pollution+'value_day.csv',index=None)
