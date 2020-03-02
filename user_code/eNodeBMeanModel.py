# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

#推后几个小时，设置预测后几个小时
HOUR_OFFSET = 1
#测试小区
test_all_cell = False
test_cell_id = set(['393805-1'])
#训练集起始
train_start = pd.to_datetime('2017-4-15')
#测试集开始
test_start = pd.to_datetime('2017-6-15')
#测试集结束
test_end = pd.to_datetime('2017-6-15 23:00:00')
#Ridge参数
ridge_alpha = 2


#相对损失
def evluation(real,pred):
    return np.abs(real-pred)/real

#绝对损失
def abs_loss(real,pred):
    return np.abs(real-pred)

#加权求和
def weight_mean(x):
    k=len(x)
    weights = np.mat(np.exp(np.array(range(len(x)))**2/(-2.0*k**2)))
    x = np.mat(x)
    x[np.isnan(x)]=0
    result = x*weights.T
    return result[0,0]

#读取数据
data = pd.read_csv('../data/month5-data.csv')
data['Time'] = pd.to_datetime(data['Time'])
data_temp = data[data['Time']>pd.to_datetime('2017-5-9')]
data = data[data['Time']<pd.to_datetime('2017-5-9')]
data1 = pd.read_csv('../data/month6-data-1.csv')
data1['Time'] = pd.to_datetime(data1['Time'])
data1 = data1[data1['Time']<pd.to_datetime('2017-6-19')]
data2 = pd.read_csv('../data/month6-data-2.csv',header=None)
data2.columns = data1.columns
data2['Time'] = pd.to_datetime(data2['Time'])
data2 = data2[data2['Time']<pd.to_datetime('2017-6-19')]
data1 = pd.concat([data1,data2])
common_cell = set(data['CellName'])&set(data1['CellName'])&set(data_temp['CellName'])

#找出数据中要用的列并挑选出数据较完整的小区
data = pd.concat([data,data1,data_temp])
data = data[['CellName','Time','L.Traffic.User.Avg', 'L.Traffic.User.Max']]#,'L.HHO.IntraeNB.IntraFreq.PrepAttOut']]
data.columns = ['cell','time','userCount','userCountMax']#,'switch']
data['flag'] = data['cell'].apply(lambda x: 1 if x in common_cell else 0)
data = data[data['flag']==1]
del data['flag']


origin_data = data.copy()

data['time'] = pd.to_datetime(data['time'])
data['enodeB'] = data['cell'].apply(lambda x:x.split('-')[0])

#筛选测试的小区
if not test_all_cell:
    test_enode_id = set()
    for i in test_cell_id:
        test_enode_id.add(i.split('-')[0])
    data = data[data['enodeB'].isin(test_enode_id)]


#基站平均值计算
temp = data.groupby(['enodeB','time'])['userCount','userCountMax'].mean().reset_index()
temp.columns = ['cell','time','userCount','userCountMax']
data = temp.copy()

data['hour'] = data['time'].apply(lambda x: x.hour)

data['userCountMax'] = np.log(data['userCountMax']+1)
data['userCount'] = np.log(data['userCount']+1)

data['weekend'] = data['time'].apply(lambda x: 1 if x.weekday()>4 else 0)
raw_data = data[['cell','time','userCount','userCountMax']].copy()


temp = raw_data.copy()
temp['time'] = temp['time']+datetime.timedelta(hours=HOUR_OFFSET-1)
#前4小时的平均用户数和最大用户数
for i in range(1,5):
    temp['time'] = temp['time']+datetime.timedelta(hours=1)
    temp.columns = ['cell','time','userCountPast'+str(i)+'hour','userCountMaxPast'+str(i)+'hour']
    data = pd.merge(data,temp,how='left',on=['cell','time'])

temp = raw_data.copy()
#前一天的平均用户数和最大用户数
temp['time'] = temp['time']+datetime.timedelta(days=1)
temp.columns = ['cell','time','userCountPast1Day','userCountMaxPast1Day']
data = pd.merge(data,temp,how='left',on=['cell','time'])


temp = raw_data.copy()
temp = temp.set_index('time')
temp = temp.sort_index()
#前2小时的滑动平均
temp_mean = temp.groupby('cell')['userCount'].rolling('2h').mean().reset_index()
temp_mean.columns = ['cell','time','rolling_mean']

temp_mean['time'] = temp_mean['time'] + datetime.timedelta(hours=HOUR_OFFSET)

data = pd.merge(data,temp_mean,how='left',on=['cell','time'])
#前2~4小时
temp_mean['time'] = temp_mean['time']+datetime.timedelta(hours=2)
temp_mean.columns = ['cell','time','rolling_mean1']
data = pd.merge(data,temp_mean,how='left',on=['cell','time'])
#前4~6小时
temp_mean['time'] = temp_mean['time']+datetime.timedelta(hours=2)
temp_mean.columns = ['cell','time','rolling_mean2']
data = pd.merge(data,temp_mean,how='left',on=['cell','time'])
#前6~8小时
temp_mean['time'] = temp_mean['time']+datetime.timedelta(hours=2)
temp_mean.columns = ['cell','time','rolling_mean3']
data = pd.merge(data,temp_mean,how='left',on=['cell','time'])
#前8~10小时
temp_mean['time'] = temp_mean['time']+datetime.timedelta(hours=2)
temp_mean.columns = ['cell','time','rolling_mean4']
data = pd.merge(data,temp_mean,how='left',on=['cell','time'])
#填补缺失值
data['rolling_mean'].fillna(data[['rolling_mean','rolling_mean1','rolling_mean2','rolling_mean3','rolling_mean4']].mean(axis=1),inplace=True)
data['rolling_mean1'].fillna(data[['rolling_mean','rolling_mean1','rolling_mean2','rolling_mean3','rolling_mean4']].mean(axis=1),inplace=True)
data['rolling_mean2'].fillna(data[['rolling_mean','rolling_mean1','rolling_mean2','rolling_mean3','rolling_mean4']].mean(axis=1),inplace=True)
data['rolling_mean3'].fillna(data[['rolling_mean','rolling_mean1','rolling_mean2','rolling_mean3','rolling_mean4']].mean(axis=1),inplace=True)
data['rolling_mean4'].fillna(data[['rolling_mean','rolling_mean1','rolling_mean2','rolling_mean3','rolling_mean4']].mean(axis=1),inplace=True)
data['rolling_mean_mean'] = data[['rolling_mean','rolling_mean1','rolling_mean2','rolling_mean3','rolling_mean4']].apply(weight_mean,axis=1)
data['tend'] = data['rolling_mean1'] - data['rolling_mean']
data['tend2'] = data[['rolling_mean2','rolling_mean3','rolling_mean4']].mean(axis=1) - data[['rolling_mean','rolling_mean1']].mean(axis=1)

temp = raw_data.copy()
temp = temp.set_index('time')
temp = temp.sort_index()
#前3小时的滑动平均
temp_mean = temp.groupby('cell')['userCount'].rolling('3h').mean().reset_index()
temp_mean.columns = ['cell','time','rolling2_mean']

temp_mean['time'] = temp_mean['time'] + datetime.timedelta(hours=HOUR_OFFSET)

data = pd.merge(data,temp_mean,how='left',on=['cell','time'])
#前3~6小时的滑动平均
temp_mean['time'] = temp_mean['time']+datetime.timedelta(hours=3)
temp_mean.columns = ['cell','time','rolling2_mean1']
data = pd.merge(data,temp_mean,how='left',on=['cell','time'])
#前6~9小时的滑动平均
temp_mean['time'] = temp_mean['time']+datetime.timedelta(hours=3)
temp_mean.columns = ['cell','time','rolling2_mean2']
data = pd.merge(data,temp_mean,how='left',on=['cell','time'])
#前9~12小时的滑动平均
temp_mean['time'] = temp_mean['time']+datetime.timedelta(hours=3)
temp_mean.columns = ['cell','time','rolling2_mean3']
data = pd.merge(data,temp_mean,how='left',on=['cell','time'])
#前12~15小时的滑动平均
temp_mean['time'] = temp_mean['time']+datetime.timedelta(hours=3)
temp_mean.columns = ['cell','time','rolling2_mean4']
data = pd.merge(data,temp_mean,how='left',on=['cell','time'])
#填补缺失值
data['rolling2_mean'].fillna(data[['rolling2_mean','rolling2_mean1','rolling2_mean2','rolling2_mean3','rolling2_mean4']].mean(axis=1),inplace=True)
data['rolling2_mean1'].fillna(data[['rolling2_mean','rolling2_mean1','rolling2_mean2','rolling2_mean3','rolling2_mean4']].mean(axis=1),inplace=True)
data['rolling2_mean2'].fillna(data[['rolling2_mean','rolling2_mean1','rolling2_mean2','rolling2_mean3','rolling2_mean4']].mean(axis=1),inplace=True)
data['rolling2_mean3'].fillna(data[['rolling2_mean','rolling2_mean1','rolling2_mean2','rolling2_mean3','rolling2_mean4']].mean(axis=1),inplace=True)
data['rolling2_mean4'].fillna(data[['rolling2_mean','rolling2_mean1','rolling2_mean2','rolling2_mean3','rolling2_mean4']].mean(axis=1),inplace=True)
data['rolling2_mean_mean'] = data[['rolling2_mean','rolling2_mean1','rolling2_mean2','rolling2_mean3','rolling2_mean4']].apply(weight_mean,axis=1)
data['tend_2'] = data['rolling2_mean1'] - data['rolling2_mean']
data['tend2_2'] = data[['rolling2_mean2','rolling2_mean3','rolling2_mean4']].mean(axis=1) - data[['rolling2_mean','rolling2_mean1']].mean(axis=1)


import warnings

warnings.filterwarnings('ignore')


raw_data = data.copy()
result = pd.DataFrame()
cell = set(data['cell'].dropna())

cellname = []
timearray = []
resultarray = []
real = []
for time in pd.date_range(test_start,test_end,freq='H'):
    print(time)
    data = raw_data[raw_data['hour']==time.hour]
    del data['hour']
    #按时间划分训练集测试集
    train = data[(data['time']<=time-datetime.timedelta(hours=HOUR_OFFSET))&(data['time']>=train_start-datetime.timedelta(hours=HOUR_OFFSET))]
    test_set = data[(data['time']==time)]
    train = train.dropna()
    for j,i in enumerate(cell):
        print(j)
        #按基站划分训练集测试集
        test_set_temp = test_set[(test_set['cell']==i)]
        train_temp = train[(train['cell']==i)]
        test_set_temp.fillna(test_set_temp.mean(),inplace=True)
        test_set_temp.fillna(0,inplace=True)
        #定义模型
        svr = Ridge(alpha=ridge_alpha)
        svr2 = SVR()
        #训练
        svr.fit(train_temp.iloc[:,5:],train_temp.iloc[:,2])
        svr2.fit(train_temp.iloc[:,5:],train_temp.iloc[:,2])
        #预测
        gpr_result = svr.predict(test_set_temp.iloc[:,5:])
        gpr_result2 = svr2.predict(test_set_temp.iloc[:,5:])
        #模型融合
        test_set_temp['predict'] = (gpr_result+gpr_result2)/2
        cellname.extend(np.array(test_set_temp['cell']))
        timearray.extend(np.array(test_set_temp['time']))
        resultarray.extend(np.array(test_set_temp['predict']))
        real.extend(np.array(test_set_temp['userCount']))

result = pd.DataFrame()
result['cell'] = cellname
result['time'] = timearray
result['predict'] = resultarray
result['predict'][result['predict']<0] = 0
result['userCount'] = real
result['loss'] = evluation(np.exp(result['userCount'])-1,np.exp(result['predict'])-1)
result['abs_loss'] = abs_loss(np.exp(result['userCount'])-1,np.exp(result['predict'])-1)

result['loss'][result['loss']==np.inf] = np.nan

print(result[['loss','abs_loss']].mean())

#融合比值模型结果
temp = pd.read_csv('../data/waveModelResult.csv')
temp.columns = ['cell','time','cell_ratio','predict_ratio']
temp['time'] = pd.to_datetime(temp['time'])
temp['enodeB'] = temp['cell'].apply(lambda x:x.split('-')[0])
result['enodeB'] = result['cell']
del result['cell']
temp = pd.merge(temp,result[['enodeB','time','predict']],how='left',on=['enodeB','time'])
temp['final'] = temp['predict_ratio'] * (np.exp(temp['predict'])-1)
temp = pd.merge(temp,origin_data[['cell','time','userCount']],how='left',on=['cell','time'])
temp['loss'] = evluation(temp['userCount'],temp['final'])
temp['abs_loss'] = abs_loss(temp['userCount'].copy(),temp['final'])
print(temp[['loss','abs_loss']].mean())
temp[['cell','time','final']].to_csv('../data/ratioResult.csv',index=False)

