# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 20:15:22 2018

@author: heaton
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
#读入原始数据
data = pd.read_csv("../data/month5/data.csv",delimiter=',',encoding='gbk')
data.drop("INDEX",axis=1,inplace=True)
#选取数据字段：小区ID，时间，平均用户数，最大用户数，并更改字段名称
data = data[['CellName','Time','L.Traffic.User.Avg','L.Traffic.User.Max']]

data['userCount'] = data['L.Traffic.User.Avg']
data['userMax'] = data['L.Traffic.User.Max']
data.drop(['L.Traffic.User.Avg','L.Traffic.User.Max'],axis=1,inplace=True)
data['Time']=data['Time'].apply(lambda x: datetime.strptime(x,"%Y/%m/%d %H:%M"))
data['hour']=data['Time'].apply(lambda x : x.hour)
temp1 = data.copy()

data = pd.read_csv("../data/month6/data-1.csv",delimiter=',',encoding='gbk')
data.drop("INDEX",axis=1,inplace=True)
data = data[['CellName','Time','L.Traffic.User.Avg','L.Traffic.User.Max']]
data['userCount'] = data['L.Traffic.User.Avg']
data['userMax'] = data['L.Traffic.User.Max']
data.drop(['L.Traffic.User.Avg','L.Traffic.User.Max'],axis=1,inplace=True)
data['Time']=data['Time'].apply(lambda x: datetime.strptime(x,"%Y/%m/%d %H:%M"))
data['hour']=data['Time'].apply(lambda x : x.hour)
temp2 = data.copy()

data = pd.read_csv("../data/month6/data-2.csv",delimiter=',',encoding='gbk')
data1= pd.read_csv("../data/month6/data-1.csv",delimiter=',',encoding='gbk')
data.columns = data1.columns
data.drop("INDEX",axis=1,inplace=True)
data = data[['CellName','Time','L.Traffic.User.Avg','L.Traffic.User.Max']]
data['userCount'] = data['L.Traffic.User.Avg']
data['userMax'] = data['L.Traffic.User.Max']
data.drop(['L.Traffic.User.Avg','L.Traffic.User.Max'],axis=1,inplace=True)
data['Time']=data['Time'].apply(lambda x: datetime.strptime(x,"%Y/%m/%d %H:%M"))
data['hour']=data['Time'].apply(lambda x : x.hour)
temp3 = data.copy()

data1 = temp1
data2 = temp2
data3 = temp3

data2 = pd.concat([data2,data3])
#根据数据选取较为连续的数据时间段
new_data1 = data1[(data1.Time>='2017-04-07') & (data1.Time<'2017-05-09')]
new_data2 = data1[(data1.Time>='2017-05-15') & (data1.Time<'2017-05-24')]
new_data3 = data2[(data2.Time>='2017-06-08') & (data2.Time<'2017-06-17')]

cell_set = set(new_data1['CellName'])&set(new_data2['CellName'])&set(new_data3['CellName'])

data = pd.concat([new_data1,new_data2,new_data3])
data['flag'] = data['CellName'].apply(lambda x: 1 if x in cell_set else 0)
data = data[data['flag'] == 1]
data.drop('flag',axis=1,inplace=True)
data.to_csv("../output/Common_CellName_new_data.csv",index=False)