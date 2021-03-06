```
广东省臭氧预报模型汇总
```

## 文件包含：

##### data：

数据组成：广东省各省市日数据文件

##### 分类模型：

**LGBM：LGBclassify.py** 用lgb模型对广东省全数据建模，2017.6.21-2020.12.31作为训练数据，2021年作为测试数据。将未来污染物的等级作为预测类别。包含特征工程，参数选择，采样技术，模型评估等内容。

**anomaly-detection.py** 。在优良和轻度以上污染类别数目样本分布不平和的情况下，用异常检测常用算法实现分类。以汕头的数据为训练和测试数据，使用LOF（局部因子分析）和IsolationForest (孤立森林)的算法实验，LOF的效果优于孤立森林。

注：此方法多在半监督任务上，训练集只有正样本，测试集中含有正、负样本。相比较LGBM，模型可以不需要大量特征工程，可通过调整参数调整正负样本比例。

**autoencoder.py**。 自编码器的方式来检测离群样本。本例子效果不佳，未深入展开。

##### 回归模型：

LGBregression.py。 

用lgbm/mlp等多种模型算法实现未来臭氧（aqi）预测， 特征包含历史的污染物特征，及未来气象特征。并保存lgbm模型

lgb_predict.py 可选择不同时间作为测试数据，可以生成各城市未来天污染物预测的csv文件

cnnlstmmul.py 

使用cnn+lstm+attention深度神经网络做时序预测，通过滑动窗口对数据进行划分。模型可以同时输出未来多日预测，数据量不足且数据划分还不够完善等问题，模型准确率较低。

##### 图：

绘制区间图和分布图绘制。

precision.py  根据预测结果文件来绘制区间图和分布图，或计算准确率。

服务器地址：/mnt/qgl/aqipredict ：aqi预测部署代码

：/mnt/qgl/o3predict ：o3预测部署代码

##### 其余：

一些用画图及其他模型测试的代码