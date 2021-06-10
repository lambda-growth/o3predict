import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve

data = pd.read_csv('../data/2020汕尾日.csv')

# 获取类别标签
def func(x):
    if x <= 100:
        return 0
    else:
        return 1

data['target'] = data['aqi'].map(func)


# data = data.loc[:,['aqi','o3_8h','wda_max', 'wda_mean', 'wda_min', 'temp_max',
#        'temp_mean', 'temp_min', 'dswrf_max', 'dswrf_mean', 'dswrf_min',
#        'humi_max', 'humi_mean', 'humi_min', 'apcp_max', 'apcp_mean',
#        'apcp_min', 'target']]

data = data.loc[:,['aqi','o3_8h', 'temp_max', 'dswrf_max',
       'humi_max', 'apcp_max', 'target']]

data.fillna(method='ffill',inplace=True)
data.fillna(method='ffill', inplace=True)
data.dropna(axis=0, inplace=True)
data_pos = data[data['target'] == 0]
data_neg = data[data['target'] == 1]

from sklearn.utils import shuffle

data_pos = shuffle(data_pos)
train = data_pos.iloc[:-50,:]
test = data_pos.iloc[-50:,:]
test = pd.concat([test,data_neg])



print(data.columns)
X_train = train[train.columns[:-1]]
y_train = train['target']

X_test = test[test.columns[:-1]]
y_test = test['target']
print(X_train.shape)
print(y_train.shape)
print(y_train.value_counts())
#数据标准化
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# 设置Autoencoder的参数
# 隐藏层节点数分别为8，4，4，8
# epoch为50，batch size为32
input_dim = X_train.shape[1]
encoding_dim = 8
num_epoch = 500
batch_size = 32

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['mae'])

# 模型保存为SofaSofa_model.h5，并开始训练模型
# checkpointer = ModelCheckpoint(filepath="train_model.h5",
#                                verbose=0,
#                                save_best_only=True)
# history = autoencoder.fit(X_train, X_train,
#                           epochs=num_epoch,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           validation_data=(X_test, X_test),
#                           verbose=1,
#                           callbacks=[checkpointer]).history
# print(history)
# # 画出损失函数曲线
# plt.figure(figsize=(14, 5))
# plt.subplot(111)
# plt.plot(history['loss'], c='dodgerblue', lw=3)
# plt.plot(history['val_loss'], c='coral', lw=3)
# plt.title('model loss')
# plt.ylabel('mse'); plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()



# 读取模型
autoencoder = load_model('train_model.h5')

# 利用训练好的autoencoder重建测试集
pred_test = autoencoder.predict(X_test)


# 计算还原误差MSE和MAE
mse_test = np.mean(np.power(X_test - pred_test, 2), axis=1)

mae_test = np.mean(np.abs(X_test - pred_test), axis=1)

mse_df = pd.DataFrame()
mse_df['Class'] = y_test
mse_df['MSE'] = mse_test
mse_df['MAE'] = mae_test
mse_df = mse_df.sample(frac=1).reset_index(drop=True)
print(mse_df)

# 分别画出测试集中正样本和负样本的还原误差MAE和MSE
markers = ['o', '^']
markers = ['o', '^']
colors = ['dodgerblue', 'coral']
labels = ['Non-fraud', 'Fraud']

plt.figure(figsize=(14, 5))
plt.subplot(121)
for flag in [0, 1]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp.index,
                temp['MAE'],
                alpha=0.7,
                marker=markers[flag],
                c=colors[flag],
                label=labels[flag])
plt.title('Reconstruction MAE')
plt.ylabel('Reconstruction MAE'); plt.xlabel('Index')
plt.subplot(122)
for flag in [0, 1]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp.index,
                temp['MSE'],
                alpha=0.7,
                marker=markers[flag],
                c=colors[flag],
                label=labels[flag])
plt.legend(loc=[1, 0], fontsize=12); plt.title('Reconstruction MSE')
plt.ylabel('Reconstruction MSE'); plt.xlabel('Index')
plt.show()

# 画出Precision-Recall曲线
plt.figure(figsize=(14, 6))
for i, metric in enumerate(['MAE', 'MSE']):
    plt.subplot(1, 2, i+1)
    precision, recall, _ = precision_recall_curve(mse_df['Class'], mse_df[metric])
    pr_auc = auc(recall, precision)
    plt.title('Precision-Recall curve based on %s\nAUC = %0.2f'%(metric, pr_auc))
    plt.plot(recall[:-2], precision[:-2], c='coral', lw=4)
    plt.xlabel('Recall'); plt.ylabel('Precision')
plt.show()

# 画出ROC曲线
plt.figure(figsize=(14, 6))
for i, metric in enumerate(['MAE', 'MSE']):
    plt.subplot(1, 2, i+1)
    fpr, tpr, _ = roc_curve(mse_df['Class'], mse_df[metric])
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic based on %s\nAUC = %0.2f'%(metric, roc_auc))
    plt.plot(fpr, tpr, c='coral', lw=4)
    plt.plot([0,1],[0,1], c='dodgerblue', ls='--')
    plt.ylabel('TPR'); plt.xlabel('FPR')
plt.show()

# 画出MSE、MAE散点图
markers = ['o', '^']
colors = ['dodgerblue', 'coral']
labels = ['Non-fraud', 'Fraud']

plt.figure(figsize=(10, 5))
for flag in [0, 1]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp['MAE'],
                temp['MSE'],
                alpha=0.7,
                marker=markers[flag],
                c=colors[flag],
                label=labels[flag])
plt.legend(loc=[0, 1])
plt.ylabel('Reconstruction RMSE'); plt.xlabel('Reconstruction MAE')
plt.show()