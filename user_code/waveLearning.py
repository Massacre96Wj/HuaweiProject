# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import GradientBoostingRegressor

#绝对损失
def abs_loss(real,pred):
    return np.abs(real-pred)

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
#GBDT参数
gbdt_trees = 100
gbdt_tree_depth = 4
gbdt_loss = 'huber'


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
data = data[['CellName','Time','L.Traffic.User.Avg', 'L.Traffic.User.Max']]
data.columns = ['cell','time','userCount','userCountMax']
data['flag'] = data['cell'].apply(lambda x: 1 if x in common_cell else 0)
data = data[data['flag']==1]
del data['flag']


#小区用户数与基站平均值比值计算
data['time'] = pd.to_datetime(data['time'])
data['enodeB'] = data['cell'].apply(lambda x:x.split('-')[0])
temp = data.groupby(['enodeB','time'])['userCount'].mean().reset_index()
temp.columns = ['enodeB','time','userCountEnodeBMean']
data = pd.merge(data,temp,how='left',on=['enodeB','time'])
data['cell_ratio'] = data['userCount'] / data['userCountEnodeBMean']
data['cell_ratio'] = data['cell_ratio'].apply(lambda x: 0 if x==np.inf else x)
data['hour'] = data['time'].apply(lambda x: x.hour)
data['weekday'] = data['time'].apply(lambda x: x.weekday())

#筛选测试的小区
if not test_all_cell:
    test_enode_id = set()
    for i in test_cell_id:
        test_enode_id.add(i.split('-')[0])
    data = data[data['enodeB'].isin(test_enode_id)]

#过去一天的小区用户数、基站下平均、比例
temp = data[['cell','time','userCount','userCountEnodeBMean','cell_ratio']].copy()
temp['time'] = temp['time'] + datetime.timedelta(days=1)
temp.columns = ['cell','time','userCountEnodeBPastDayMean','userCountPastDay','cell_ratioPastDay']
data = pd.merge(data,temp,how='left',on=['cell','time'])

temp = data[['cell','time','userCount','userCountEnodeBMean','cell_ratio']].copy()
temp['time'] = temp['time']+datetime.timedelta(hours=HOUR_OFFSET-1)

#前4小时的基站平均、平均用户数和用户数比值
for i in range(1,5):
    temp['time'] = temp['time']+datetime.timedelta(hours=1)
    temp.columns = ['cell','time','userCountEnodeBPast'+str(i)+'HourMean','userCountPast'+str(i)+'Hour','cell_ratioPast'+str(i)+'Hour']
    data = pd.merge(data,temp,how='left',on=['cell','time'])

result = pd.DataFrame()
for time in pd.date_range(test_start,test_end,freq='H'):
    print(time)
    #按时间划分训练集测试集
    train = data[(data['time']<=time-datetime.timedelta(hours=HOUR_OFFSET))&(data['time']>=train_start-datetime.timedelta(hours=HOUR_OFFSET))]
    train = train.dropna()
    if(train.empty): continue
    test = data[data['time']==time]
    for j,i in enumerate(set(data['enodeB'])):
        print(j)
        #按基站划分训练集测试集
        trainTemp = train[train['enodeB']==i]
        testTemp = test[test['enodeB']==i]
        testTemp.fillna(testTemp.mean(),inplace=True)
        testTemp.fillna(0,inplace=True)
        #定义模型
        svr = GradientBoostingRegressor(n_estimators=gbdt_trees,max_depth=gbdt_tree_depth,loss=gbdt_loss)
        #训练
        svr.fit(trainTemp.iloc[:,7:],trainTemp.iloc[:,6])
        #预测
        svrResult = svr.predict(testTemp.iloc[:,7:])
        testTemp['predict'] = svrResult
        result = pd.concat([result,testTemp])

result = result[['cell','time','cell_ratio','predict']]
result.to_csv('../data/waveModelResult.csv',index=False)
print(abs_loss(result['cell_ratio'],result['predict']).mean())
