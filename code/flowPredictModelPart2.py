# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 12:09:46 2017

@author: heaton
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import pandas as pd
from scipy.stats import boxcox
from dateutil.parser import parse
starttime = datetime.now()





def get_before_hour_sub(data,from_data,timeStr,hour_,feature_):
    """
    data:训练集
    from_data:特征提取集合.当天
    timeStr:当天集合
    程序主要是为了处理0 点，1点这种特殊时刻
    """
    timeStr_before=(datetime.strptime(timeStr,"%Y-%m-%d")-timedelta(1)).strftime("%Y-%m-%d")
    fea_name="Before"+str(hour_)+"Hour_"+feature_
    last_day_hour=24-hour_
    
    from_data1=from_data[(from_data.Time==timeStr_before)&(from_data.hour>=last_day_hour)].copy()
    from_data1=from_data1[["CellName","hour",feature_]].copy()
    from_data1['hour']=from_data1['hour']-last_day_hour
    from_data1= from_data1.rename(columns={feature_:fea_name})

    data_copy=pd.DataFrame.copy(data,deep=True)
    data_copy['hour']=data_copy['hour']+hour_

    data_copy_new=data_copy[["CellName","hour",feature_]].copy()
    data_copy_new=data_copy_new.rename(columns={feature_:fea_name})
    data=pd.merge(data,data_copy_new,on=["CellName","hour"],how='outer')
    
    data_hour_0=data[(data.hour<hour_)&(data.hour>=0)]
    data_hour_no_0=data[(data.hour>=hour_)&(data.hour<24)]
   
    data_hour_0=data_hour_0.drop([fea_name],axis=1)
    data_hour_0=pd.merge(data_hour_0,from_data1,on=["CellName","hour"],how='outer')
    
    data=pd.concat([data_hour_0,data_hour_no_0])
    data=data[["CellName","hour",fea_name]].copy()
    
    data=data.fillna(0)
    #print(data)
    return data
def get_before_hour_1(data,timeStr,fea_name,hour_index):
    """
    得到前几个小时的userCount
    timeStr 当前时间
    给to_data
    """
    data=data[['CellName','hour','Time',fea_name]]
    to_data=data[data.Time==date_used[date_index]].copy()
    before_day=(datetime.strptime(date_used[date_index],"%Y-%m-%d")-timedelta(1)).strftime("%Y-%m-%d")
    from_data=data[data.Time==before_day]
    
    res=to_data[['CellName','hour']].copy()
    from_data=from_data[['CellName','hour','Time',fea_name]].copy()
    
    for hour_ in hour_index:
        tmp=get_before_hour_sub(to_data,from_data,timeStr,hour_,fea_name)
        #print(tmp)
        res=pd.merge(res,tmp,on=["CellName","hour"],how='outer')
    #print(" hehe ")
    
    res=pd.merge(res,to_data,on=['CellName','hour'],how='outer')
    
    #print( "do res ")
    return res
#提取对应时间前1，2，3，4小时特征
#ULThQci：前1，2，3，4
#ULThQciMax：前1
#ULThQci6：前1
#ULThQci6Max：前1
#userCount：前1,2,3,4
def get_before_hour(data,timeStr,HOUR_OFFSET):
    hour_begin=HOUR_OFFSET
    fea_col=['ULThQci','ULThQciMax','ULThQci6','ULThQci6Max','userCount']
             
    hour_index=[[hour_begin,hour_begin+1,hour_begin+2,hour_begin+3],[hour_begin],[hour_begin],[hour_begin],[hour_begin]]
    result=pd.DataFrame()
    for fea_index in range(len(fea_col)):
        tmp=get_before_hour_1(data,timeStr,fea_col[fea_index],hour_index[fea_index])
        if np.shape(result)[0]==0:
            result=tmp
        else:
            result=pd.merge(result,tmp,on=['CellName','Time','hour'],how='outer')
    return result

#提取对应时间前1天特征	
def get_before_day(data,timeStr):
    timeStr_before=(datetime.strptime(timeStr,"%Y-%m-%d")-timedelta(1)).strftime("%Y-%m-%d")
    data_before=data[data.Time==timeStr_before].copy()
    data_before.rename(columns={'ULThQci':'before1d_DL'},inplace=True)
    data_=data[data.Time==timeStr].copy()
    data_=data_[["CellName","hour","Time"]].copy()
    data_before=data_before[["CellName","hour",'before1d_DL']].copy()
    data_=pd.merge(data_,data_before,on=['CellName','hour'],how='outer')
    return data_

#提取对应时间前3天特征	
def get_3before_day(data,timeStr):
    timeStr_before=(datetime.strptime(timeStr,"%Y-%m-%d")-timedelta(3)).strftime("%Y-%m-%d")
    data_before=data[data.Time>=timeStr_before]
    data_before = data_before[data_before.Time<timeStr]
    data_before1 = data_before.groupby(by=['CellName','hour'],as_index=False)['ULThQci'].mean()
    data_before1.rename(columns={'ULThQci':'before1d_3DLMean'},inplace=True)
    data_before2 = data_before.groupby(by=['CellName','hour'],as_index=False)['ULThQci'].median()
    data_before2.rename(columns={'ULThQci':'before1d_3DLMedian'},inplace=True)

    data_before3 = data_before.groupby(by=['CellName','hour'],as_index=False)['userCount'].mean()
    data_before3.rename(columns={'userCount':'userCountbefore3DLMean'},inplace=True)
    data_before4 = data_before.groupby(by=['CellName','hour'],as_index=False)['userCount'].median()
    data_before4.rename(columns={'userCount':'userCountbefore3DLMedian'},inplace=True)

    
    data_=data[data.Time==timeStr].copy()
    data_=data_[["CellName","hour","Time"]].copy()
    data_before1=data_before1[["CellName","hour",'before1d_3DLMean']].copy()
    data_before2=data_before2[["CellName","hour",'before1d_3DLMedian']].copy()
    data_before3=data_before3[["CellName","hour",'userCountbefore3DLMean']].copy()
    data_before4=data_before4[["CellName","hour",'userCountbefore3DLMedian']].copy()    
    data_=pd.merge(data_,data_before1,on=['CellName','hour'],how='outer')
    data_=pd.merge(data_,data_before2,on=['CellName','hour'],how='outer')
    data_=pd.merge(data_,data_before3,on=['CellName','hour'],how='outer')
    data_=pd.merge(data_,data_before4,on=['CellName','hour'],how='outer')

    return data_


def deal_missing_data_sub(data,fea_name):
    """
    处理缺失值，根据对应小区对应不同时刻的数据中位值进行填充
    """
    data=data[['CellName','Time','hour',fea_name]].copy()
    CellName_id=set(data.CellName)
    date_hour_range=pd.date_range((parse(dateStart)-timedelta(1)).strftime("%m/%d/%Y")+" 00:00",(parse(dateTestStart)-timedelta(1)).strftime("%m/%d/%Y")+" 23:00",freq='H')
    
    date_CellName=pd.DataFrame()
    for CellName_id_index in CellName_id:
        tmp=pd.DataFrame()
        tmp['Time']=date_hour_range
        tmp['CellName']=CellName_id_index
        date_CellName=pd.concat([date_CellName,tmp],axis=0)
    date_CellName['hour']=date_CellName['Time'].apply(lambda x : x.hour)
    date_CellName['Time']=date_CellName['Time'].apply(lambda x: str(x.date()))
    data=pd.merge(data,date_CellName,on=['CellName','Time','hour'],how='outer')
    
    data_null=data[data[fea_name].isnull()].copy()
    data_null=data_null.drop([fea_name],axis=1)
    
    data_notnull=data[data[fea_name].notnull()].copy()
    
    data_median=data_notnull.groupby(['CellName','hour'])[fea_name].median().reset_index()
    
    data_null=pd.merge(data_median,data_null,on=['CellName','hour'],how='inner')
    data_new=pd.concat([data_null,data_notnull],axis=0)
    return data_new

def deal_missing_data(data):
    fea_col=['ULThQci','ULThQciMax','ULThQci6','ULThQci6Max','userCount']
    result=pd.DataFrame()
    for fea in fea_col:
        if np.shape(result)[0]==0:
            result=deal_missing_data_sub(data,fea)
        else:
            result=pd.merge(result,deal_missing_data_sub(data,fea),on=['CellName','Time','hour'],how='outer')
    return result


#参数设置
#训练集日期选取
dateStart = "4/15/2017"
#测试集起止时间
dateTestStart = "6/15/2017"
dateTestEnd = "6/15/2017"
#设置预测后几个小时
HOUR_OFFSET = 1
#选取数据集合
dateTrainUsed = []
#根据公司所提供数据当前选取较为完整的三段数据
dateSelect = [pd.date_range('2017-04-15','2017-05-07'),
              pd.date_range('2017-05-17','2017-05-22'),
              pd.date_range('2017-06-09','2017-06-15')]
for part in dateSelect:
    for daySelect in part :
        dateTrainUsed.append(daySelect.strftime("%Y-%m-%d"))
#是否进行小区级测试
test_all_cell =True
test_cell_id = set(['393766-4'])  
#是否进行小时级测试
test_all_hour =True
test_hour_list = [8]    
#LGBM可设置参数：
#lightgbm.LGBMModel(boosting_type='gbdt',
                    # num_leaves=31,
                    # max_depth=-1,
                    # learning_rate=0.1,
                    # n_estimators=10,
                    # max_bin=255,
                    # subsample_for_bin=200000, 
                    # objective=None,
                    # min_split_gain=0.0, 
                    # min_child_weight=0.001, 
                    # min_child_samples=20, 
                    # subsample=1.0, 
                    # subsample_freq=1,
                    # colsample_bytree=1.0,
                    # reg_alpha=0.0, 
                    # reg_lambda=0.0, 
                    # random_state=None, 
                    # n_jobs=-1,
                    # silent=True, **kwargs)
#本部分模型具体设置如下参数：
n_estimators=400
max_depth=3
learning_rate=0.01   
#数据平滑系数：
rate =1.5   
#模型融合比：
a = 0.4
b = 0.6
#读取流量数据，用户数据
data=pd.read_csv("../output/Common_CellName_flow_data.csv")
user_data=pd.read_csv("../output/Common_CellName_new_data.csv")        
#训练集日期选取
date_used=pd.date_range(dateStart,dateTestEnd)
date_used=[x.strftime("%Y-%m-%d") for x in date_used]
#合并训练数据和用户数据，选取所用字段
user_data= user_data[['CellName','Time','userCount']]
data = pd.merge(data,user_data,on=['CellName','Time'],how='left')
data = data[['CellName','Time','hour','ULThQci','ULThQciMax','ULThQci6',
             'ULThQci6Max','userCount']]
#筛选测试的小区
if not test_all_cell:
    data = data[data['CellName'].isin(test_cell_id)]
    
data.Time = pd.to_datetime(data.Time)
data['Time']=data['Time'].apply(lambda x: x.strftime("%Y-%m-%d"))
#缺失值处理
data=deal_missing_data(data)

data_ts=pd.DataFrame()
sp_date=set()
#特征提取，合并特征字段
for date_index  in range(len(date_used)):
    if date_used[date_index] in sp_date:
        continue       
    res=get_before_hour(data,date_used[date_index],HOUR_OFFSET)
    res2=get_before_day(data,date_used[date_index])
    res3=get_3before_day(data,date_used[date_index])
    res=pd.merge(res,res2,on=['CellName','Time','hour'],how='outer')
    res=pd.merge(res,res3,on=['CellName','Time','hour'],how='outer')
    if date_index==0:
        data_ts=res
    else:
        data_ts=pd.concat([data_ts,res],axis=0)
        
#训练集合日期选取
data_ts = data_ts[data_ts['Time'].isin(dateTrainUsed)]


data_ts.sort_values(by=['CellName','hour'],inplace=True)
temp = pd.get_dummies(data_ts['hour'])
data_ts = pd.concat([data_ts,temp],axis=1)
cell_id=list(set(data.CellName)) 

date_test=pd.date_range(dateTestStart,dateTestEnd)
date_test=[x.strftime("%Y-%m-%d") for x in date_test]
date_train=pd.date_range(dateStart,(parse(dateTestStart)-timedelta(1)).strftime("%m/%d/%Y"))
date_train=[x.strftime("%Y-%m-%d") for x in date_train]

model_dict={}


predict_result=[]
for day in date_test:
    predict_val=np.zeros(len(cell_id))
    real_x=np.zeros(len(cell_id))
    train_data=data_ts[data_ts.Time<day].copy()#训练集合
    test_data=data_ts[data_ts.Time==day].copy()#测试集合
    if not test_all_hour:
        test_data = test_data[test_data['hour'].isin(test_all_hour)]
    temp = train_data.groupby(by=['CellName','hour'],as_index=False)['ULThQci'].mean()
    temp = temp.rename(columns={'ULThQci':'ULThQci_mean'})
    train_data = pd.merge(train_data,temp,on=['CellName','hour'],how='left')
    indexHigh = train_data[train_data['ULThQci']>train_data['ULThQci_mean']*rate].index
    train_data.ix[indexHigh,'ULThQci'] = train_data.ix[indexHigh,'ULThQci_mean'] *rate
                                  
    train_data = train_data.drop(['ULThQci_mean'],axis=1)          
    for cell_id_index in range(len(cell_id)):#小区ID
        lr1 = LGBMRegressor(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate) #模型LGB 
        train_data_tmp=train_data[train_data.CellName==cell_id[cell_id_index]].copy()
        train_data_tmp.drop(['CellName','Time'],axis=1,inplace=True)
        train_x=train_data_tmp.drop([
        'ULThQci','ULThQciMax','ULThQci6','ULThQci6Max','userCount'],axis=1)
        label = np.log(train_data_tmp['ULThQci']+1)#label进行log变换
        lr1.fit(train_x,label)
        test_data_tmp=test_data[test_data.CellName==cell_id[cell_id_index]].copy()
        test_x=test_data_tmp.drop(['CellName','Time','ULThQci','ULThQciMax',          
             'ULThQci6','ULThQci6Max','userCount'],axis=1)   
        pre_y1=lr1.predict(test_x)#模型预测
        pre_y1 = [np.e**x-1 for x in pre_y1]#log变换还原
        test_data_tmp['LGB'] = [max(x,0) for x in pre_y1]
        predict_result.append(test_data_tmp)
predict_result_= pd.concat(predict_result) 
result_no_0=predict_result_[predict_result_.ULThQci>0].copy()
result_no_0['predict_val'] = result_no_0['LGB']
result_no_0['absolute error']=result_no_0['ULThQci']-result_no_0['predict_val']
result_no_0['absolute error']=result_no_0['absolute error'].apply(lambda x :abs(x))
result_no_0['relative error']=result_no_0['absolute error']/result_no_0['ULThQci']
print("整体建模算法误差门限为0误差为 ：",np.mean(result_no_0.loc[result_no_0.ULThQci>0,'relative error']))
result_no_0_part = result_no_0[result_no_0.ULThQci>0] 
result_no_0_part = result_no_0_part.groupby(by=['Time'],as_index=False)['relative error'].mean()
print(result_no_0_part) 
endtime = datetime.now()
print(endtime-starttime) 

result_no_0.to_csv("../output/modelResSec.csv",index=False)#第二部分模型预测结果

dataCombine = pd.read_csv("../output/modelResFir.csv")#读取第一部分模型预测结果
  
dataCombine = pd.merge(dataCombine,result_no_0,on=['CellName','hour','Time'])#融合第一部分预测结果，第二部分预测结果
dataCombine['pre'] = dataCombine['predict_val_x']*a+dataCombine['predict_val_y']*b
dataCombine['absolute error']=dataCombine['ULThQci']-dataCombine['pre']
dataCombine['absolute error']=dataCombine['absolute error'].apply(lambda x :abs(x))
dataCombine['relative error']=dataCombine['absolute error']/dataCombine['ULThQci']
print("融合模型算法误差门限为0误差为 ：")
print(np.mean(dataCombine.loc[dataCombine.ULThQci>0,'relative error']))
dataCombineUsed = dataCombine[dataCombine.ULThQci>0] 
dataCombineUsed = dataCombineUsed.groupby(by=['Time'],as_index=False)[['relative error']].mean()
print(dataCombineUsed) 

result_Final = dataCombine[['CellName','Time','hour','pre','real']]
result_Final.to_csv('../output/resultFinal.csv',index=False)
