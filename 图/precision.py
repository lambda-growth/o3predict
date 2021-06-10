import pandas as pd
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import base
citylist = base.area_group('广东')



def precision_rate(pollution):
#准确率
    precision_list = []
    # for city in citylist:
    for city in ['广州']:
        df = pd.read_csv('./'+pollution+'预测/2'+city+'predict_'+pollution+'value_day.csv')
        # df = pd.read_csv('./'+pollution+'预测/aa.csv')
        predicted_value = df.iloc[:, 2]
        # predicted_value = df.iloc[:, 1:7].mean(axis=1)
        Y = df.iloc[:,-1]
        precision = 0
        for i in range(len(predicted_value)):
            if Y[i] -  25 <= predicted_value[i] <= Y[i] + 25:
                precision += 1

        precision_rate = precision / len(predicted_value)
        precision_list.append(precision_rate)
        print('city: {}, precision_rate：{}'.format(city, precision_rate))
    precision = pd.DataFrame({'city':citylist, 'precision':precision_list})
    # precision.to_csv('./'+pollution+'预测/o3precision.csv')
    print('mean:',np.array(precision_list).mean())



def plot_line(df,city,pollution):

    date2_1 = datetime.datetime(2021, 1, 1)
    date2_2 = datetime.datetime(2021, 5, 8)
    delta2 = datetime.timedelta(days=1)
    xtime = mpl.dates.drange(date2_1, date2_2, delta2)
    df['time'] = xtime
    # df['time'] = df['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
    # df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"))
    y_pre = df.iloc[:, 2]
    # y_pre2 = df.iloc[:55, 2]
    y_real = df.iloc[:, -1]
    x = df.iloc[:, 0]

    y1 = list(y_real - 15)
    y2 = list(y_real + 15)
    plt.rcParams['figure.figsize'] = (12.0, 5.0)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.fill_between(x, y1, y2, alpha=.08, color='b')
    # ax1.set_ylabel(city + pollution+'值', fontsize='15')
    ax1.plot(x, y_pre, '#FFA340', ms=10)
    # ax1.plot(x, y_pre2, 'red', ms=10)
    ax1.plot(x, y_real, '#58aa72', ms=10)
    ax1.set_xlabel('time', fontsize='15')
    ax1.legend([ "预测值","真实值"], loc='upper right')
    # 设置横坐标格式
    dateFmt = mpl.dates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(dateFmt)
    daysLoc = mpl.dates.DayLocator(interval=10)
    plt.xticks(rotation=45)
    ax1.xaxis.set_major_locator(daysLoc)
    plt.savefig('./区间图/'+pollution+'/gai2'+city+pollution+'区间预测图.png',bbox_inches = 'tight')
    # plt.show()
    plt.close()

 # 画区间图

def plot_gap(df,city,pollution):
    date2_1 = datetime.datetime(2021, 1, 1)
    date2_2 = datetime.datetime(2021, 5, 8)
    delta2 = datetime.timedelta(days=1)
    xtime = mpl.dates.drange(date2_1, date2_2, delta2)
    df['time'] = xtime
    # df['time'] = df['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
    # df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"))
    y_pre = df.iloc[:, 1]
    # y_pre2 = df.iloc[:55, 2]
    y_real = df.iloc[:, -1]
    x = df.iloc[:, 0]
# 画柱形折现组合图
    plt.rcParams['figure.figsize'] = (12.0,5.0)
    fig = plt.figure()
    #画柱形图
    ax1 = fig.add_subplot(111)
    ax1.bar(x, y_real,alpha=.7,color='#58aa72')
    res = max(y_real)
    print(res)
    ax1.set_ylim([0,250])
    # ax1.set_ylabel(city+pollution+'值',fontsize='15')
    ax1.legend(["真实值"],loc = 'upper center')

    #ax1.set_title("数据统计",fontsize='20')
    #画折线图
    ax2 = ax1.twinx()   #组合图必须加这个
    ax2.plot(x, y_pre, '#FFA340',ms=10)
    # ax2.plot(x, y_pre2, 'red', ms=10)

    ax2.set_xlabel('time',fontsize='15')
    ax2.set_ylim([0,250])
    # ax2.legend(["未加入时间信息预测值", "加入时间信息预测值","真实值"], loc='upper right')
    ax2.legend(["预测值"], loc='upper right')
    plt.gcf().autofmt_xdate()
    dateFmt = mpl.dates.DateFormatter('%Y-%m-%d')
    plt.gca().xaxis.set_major_formatter(dateFmt)

    daysLoc = mpl.dates.DayLocator(interval=10)
    plt.xticks(rotation=270)
    plt.gca().xaxis.set_major_locator(daysLoc)
    plt.savefig('./折现柱形图/'+pollution+'/gai3'+city+pollution+'组合图.png')

    # ax2.set_xticklabels(x)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
    # plt.show()
    plt.close()
if __name__ == '__main__':

    pollution = 'o3'
    # for city in citylist:
    for city in ['广州']:
        df = pd.read_csv('./'+pollution+'预测/2'+city+'predict_o3value_day.csv')
        # plot_gap(df,city,pollution)
        plot_line(df,city,pollution)
    # precision_rate(pollution)