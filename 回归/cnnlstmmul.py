import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM, Activation,Conv1D,MaxPooling1D,Flatten,GRU
from tensorflow.keras.layers import RepeatVector, TimeDistributed
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(tf.__version__)
#周边城市潮州，揭阳市,
data = pd.read_csv('../data/2020汕头日.csv')
print(data.columns)
data.drop_duplicates(subset=['time'],inplace=True)
data.dropna(axis=0,subset = ['aqi', 'cityname',  'co', 'no2', 'so2', 'o3_8h',
       'pm2_5', 'pm10', 'wda_max', 'wda_mean', 'wda_min', 'temp_max',
       'temp_mean', 'temp_min', 'dswrf_max', 'dswrf_mean', 'dswrf_min',
       'humi_max', 'humi_mean', 'humi_min', 'apcp_max', 'apcp_mean',
       'apcp_min'],inplace=True)
for idx,city in enumerate(data['cityname'].unique()):
    data1 = data.drop(data[data.cityname!=city].index)
    print(len(data1))
    data1 = data1.loc[:, ['aqi', 'co', 'no2', 'so2', 'o3_8h',
       'pm2_5', 'pm10', 'wda_max', 'wda_mean', 'wda_min', 'temp_max',
       'temp_mean', 'temp_min', 'dswrf_max', 'dswrf_mean', 'dswrf_min',
       'humi_max', 'humi_mean', 'humi_min', 'apcp_max', 'apcp_mean',
       'apcp_min']]
    if idx == 0:
        training_set = data1.iloc[1:-300, :].values
        test_set = data1.iloc[-300:, :].values
    else:
        training_set2 = data1.iloc[1:, :].values
        training_set = np.append(training_set, training_set2, axis=0)




def create_dataset(data,n_predictions,n_next):
    '''
    n_predictions:前面步数
    n_next: 预测后面步数
    对数据进行处理
    '''
    train_X, train_Y = [], []
    for i in range(data.shape[0]-n_predictions-n_next-1):
        a = data[i:(i+n_predictions),:]
        train_X.append(a)
        tempb = data[(i+n_predictions):(i+n_predictions+n_next),0]
        b = []
        for j in range(len(tempb)):
            b.append(tempb[j])
        train_Y.append(b)
    train_X = np.array(train_X,dtype='float64')
    train_Y = np.array(train_Y,dtype='float64')

    return train_X, train_Y

from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras.layers import Layer

class Attention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, hidden_states):
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28.
        """
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

def trainModel(train_X, train_Y):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    '''

    #第一层卷积读取输入序列，并将结果投影到特征图上。第二层卷积在第一层创建的特征图上执行相同的操作，
    # 尝试放大其显著特征。每个卷积层使用64个特征图（filters=64），并以3个时间步长的内核大小（kernel_size=3）读取输入序列。
    # 最大池化层降采样成原来特征图尺寸的1/4来简化特征图。然后将提取的特征图展平为一个长向量，将其用作解码过程的输入。
    model = tf.keras.Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu',#一维CNN能够读取序列输入并自动学习显着特征50-3+1 = 48
               input_shape=(train_X.shape[1], train_X.shape[2])),
        # Conv1D(filters=64, kernel_size=3, activation='relu'),
        # MaxPooling1D(pool_size=2),
        # Flatten(),
        # RepeatVector(train_Y.shape[1]),
        GRU(80, activation='relu',return_sequences=True),#输入（batch_size, timestamp, feature)
        Dropout(0.2),
        # GRU(100, activation='relu',return_sequences=True),#输入（batch_size, timestamp, feature)
        # Dropout(0.3),
        Attention(),
        Dense(80, activation='relu'),
        Dense(3)
    ])

    # model = tf.keras.Sequential([
    #     Conv1D(filters=64, kernel_size=3, activation='relu',  # 一维CNN能够读取序列输入并自动学习显着特征50-3+1 = 48
    #            input_shape=(train_X.shape[1], train_X.shape[2])),
    #     Conv1D(filters=64, kernel_size=3, activation='relu'),
    #     # MaxPooling1D(pool_size=2),
    #     Flatten(),
    #     RepeatVector(train_Y.shape[1]),
    #     LSTM(200, activation='relu', return_sequences=True),  # 输入（batch_size, timestamp, feature)
    #     Dropout(0.2),
    #     TimeDistributed(Dense(100, activation='relu')),
    #     TimeDistributed(Dense(1))
    # ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='mean_squared_error')
    return model

# 归一化

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range,np.max(data),np.min(data)

def transform(data):
    return (data - data_min) / (data_max-data_min)
def inverse_transform(data):
    return data * (data_max-data_min) + data_min

sc = StandardScaler()  # 定义归一化：归一化到(0，1)之间
training_set_scaled,data_max,data_min = normalization(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
test_set_scaled = transform(test_set)
print(test_set_scaled)
# 利用训练集的属性对测试集进行归一化

train_X, train_Y = create_dataset(training_set_scaled, 10, 3)
# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(train_X)
np.random.seed(7)
np.random.shuffle(train_Y)
tf.random.set_seed(7)
# 将训练集由list格式变为array格式
train_X, train_Y = np.array(train_X), np.array(train_Y)
test_X, test_Y = create_dataset(test_set_scaled, 10, 3)
print(train_X.shape,test_Y.shape)
def train():
    #模型
    model = trainModel(train_X, train_Y)
    checkpoint_save_path = "../checkpoint/cnnLSTM_5d_o3.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     monitor='val_loss')
    history = model.fit(train_X, train_Y, batch_size=64, epochs=100, validation_split = 0.25, validation_freq=1,
                    callbacks=[cp_callback])
    print(model.summary())

    file = open('../weights.txt', 'w')  # 参数提取
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

################## predict ######################
def predict():
    # 测试集输入模型进行预测
    #模型

    model = trainModel(train_X, train_Y)
    checkpoint_save_path = "./checkpoint/cnnLSTM_5d_o3.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)
    predicted_value = model.predict(test_X)

    predicted_value = tf.squeeze(input=predicted_value)


    real_value = inverse_transform(test_Y)
    # 对预测数据还原---从（0，1）反归一化到原始范围
    predicted_value = inverse_transform(predicted_value)
    # 对真实数据还原---从（0，1）反归一化到原始范围

    # 画出真实数据和预测数据的对比曲线
    print(predicted_value[1],real_value[1])
    plt.plot(real_value, color='red', label='Real o3 value')
    plt.plot(predicted_value, color='blue', label='Predicted o3 value')
    plt.title('o3 Prediction')
    plt.xlabel('Time')
    plt.ylabel('o3 value')
    plt.legend()
    plt.show()
    print(predicted_value)
    ##########evaluate##############
    # calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
    mse = mean_squared_error(predicted_value, real_value)
    # calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
    rmse = math.sqrt(mean_squared_error(predicted_value, real_value))
    # calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
    mae = mean_absolute_error(predicted_value, real_value)
    print('均方误差: %.6f' % mse)
    print('均方根误差: %.6f' % rmse)
    print('平均绝对误差: %.6f' % mae)
    predicted_value = predicted_value.numpy()
    def func(x):
        if x <= 100:
            return '优'
        elif x < 160:
            return '良'
        else:
            return '轻度污染'
    # for i in predicted_value:
    #     for j in i:
    #         if test_Y*0.9 < j < test_Y*1.1:
    precision,rank_eq  = 0, 0
    print(predicted_value.shape)
    predicted_value = predicted_value.flatten()
    real_value = real_value.flatten()
    print(predicted_value.shape)
    for i in range(len(predicted_value)):
        if real_value[i]-15 <= predicted_value[i] <= real_value[i] +15:
            precision += 1
        if list(map(func, predicted_value))[i] == list(map(func, real_value))[i]:
            rank_eq += 1
    # for i in range(len(predicted_value)):
    #     for j in range(len(predicted_value[0])):
    #         if real_value[i][j] * 0.8 <= predicted_value[i][j] <= real_value[i][j]*1.2:
    #             predicted_value[i][j]
    #             precision+=1
    #         if list(map(func, predicted_value))[i] == list(map(func, real_value))[i]:
    #             rank_eq += 1

    precision_rate = precision/len(predicted_value)
    rank_precision_rate = rank_eq / len(predicted_value)

    print('准确率: %.6f' % precision_rate)
    print('优良等级准确率: %.6f' % rank_precision_rate)


if __name__ == '__main__':
    train()
    predict()