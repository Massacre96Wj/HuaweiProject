# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 18:16:27 2018

@author: heaton
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
#读入原始数据
data = pd.read_csv("../data/month5/data.csv",delimiter=',',encoding='gbk')

data1 = pd.read_csv("../data/month6/data-1.csv",delimiter=',',encoding='gbk')
data2 = pd.read_csv("../data/month6/data-2.csv",delimiter=',',encoding='gbk')
data2.columns = data1.columns
data1 = pd.concat([data1,data2])
data.Time = pd.to_datetime(data.Time)
data1.Time = pd.to_datetime(data1.Time)
#选取可用日期段
#选取数据集合
dateTrainUsed = []
#根据公司所提供数据当前选取较为完整的三段数据
new_data1 = data[(data.Time>='2017-04-07') & (data.Time<'2017-05-09')]
new_data2 = data[(data.Time>='2017-05-15') & (data.Time<'2017-05-24')]
new_data3 = data1[(data1.Time>='2017-06-08') & (data1.Time<'2017-06-17')]

cell_set = set(new_data1['CellName'])&set(new_data2['CellName'])&set(new_data3['CellName'])
#合并数据
new_data = pd.concat([new_data1,new_data2,new_data3])
new_data['flag'] = new_data['CellName'].apply(lambda x: 1 if x in cell_set else 0)
final_data = new_data[new_data['flag'] == 1]
final_data.drop('flag',axis=1,inplace=True)

final_data.drop("INDEX",axis=1,inplace=True)

final_data = final_data[['CellName','Time','L.Thrp.bits.DL(bit)','L.Thrp.bits.DL.Max(bit)',
             'L.Thrp.bits.DL.QCI.5(bit)','L.Thrp.bits.DL.QCI.5.Max(bit)',
             'L.Thrp.bits.DL.QCI.6(bit)','L.Thrp.bits.DL.QCI.6.Max(bit)',
             'L.Thrp.bits.DL.QCI.7(bit)','L.Thrp.bits.DL.QCI.7.Max(bit)',
             'L.Thrp.bits.DL.QCI.8(bit)','L.Thrp.bits.DL.QCI.8.Max(bit)',
             'L.Thrp.bits.DL.QCI.9(bit)','L.Thrp.bits.DL.QCI.9.Max(bit)',             
             'L.Thrp.bits.UL(bit)','L.Thrp.bits.UL.Max(bit)',
             'L.Thrp.bits.UL.QCI.5(bit)','L.Thrp.bits.UL.QCI.5.Max(bit)',              
             'L.Thrp.bits.UL.QCI.6(bit)','L.Thrp.bits.UL.QCI.6.Max(bit)',             
             'L.Thrp.bits.UL.QCI.7(bit)','L.Thrp.bits.UL.QCI.7.Max(bit)',
             'L.Thrp.bits.UL.QCI.8(bit)','L.Thrp.bits.UL.QCI.8.Max(bit)',
             'L.Thrp.bits.UL.QCI.9(bit)','L.Thrp.bits.UL.QCI.9.Max(bit)']]             
                            
new_data = final_data[['L.Thrp.bits.DL(bit)','L.Thrp.bits.DL.Max(bit)',
             'L.Thrp.bits.DL.QCI.5(bit)','L.Thrp.bits.DL.QCI.5.Max(bit)',
             'L.Thrp.bits.DL.QCI.6(bit)','L.Thrp.bits.DL.QCI.6.Max(bit)',
             'L.Thrp.bits.DL.QCI.7(bit)','L.Thrp.bits.DL.QCI.7.Max(bit)',
             'L.Thrp.bits.DL.QCI.8(bit)','L.Thrp.bits.DL.QCI.8.Max(bit)',
             'L.Thrp.bits.DL.QCI.9(bit)','L.Thrp.bits.DL.QCI.9.Max(bit)',             
             'L.Thrp.bits.UL(bit)','L.Thrp.bits.UL.Max(bit)',
             'L.Thrp.bits.UL.QCI.5(bit)','L.Thrp.bits.UL.QCI.5.Max(bit)',              
             'L.Thrp.bits.UL.QCI.6(bit)','L.Thrp.bits.UL.QCI.6.Max(bit)',             
             'L.Thrp.bits.UL.QCI.7(bit)','L.Thrp.bits.UL.QCI.7.Max(bit)',
             'L.Thrp.bits.UL.QCI.8(bit)','L.Thrp.bits.UL.QCI.8.Max(bit)',
             'L.Thrp.bits.UL.QCI.9(bit)','L.Thrp.bits.UL.QCI.9.Max(bit)']]
#列名更正
for i in new_data.columns:
    final_data[i] = final_data[i]/(1024*1024*8)
    
final_data['DLThQci'] = final_data['L.Thrp.bits.DL(bit)']
final_data['DLThQciMax'] = final_data['L.Thrp.bits.DL.Max(bit)']
final_data['DLThQci5'] = final_data['L.Thrp.bits.DL.QCI.5(bit)']
final_data['DLThQci5Max'] = final_data['L.Thrp.bits.DL.QCI.5.Max(bit)']
final_data['DLThQci6'] = final_data['L.Thrp.bits.DL.QCI.6(bit)']
final_data['DLThQci6Max'] = final_data['L.Thrp.bits.DL.QCI.6.Max(bit)']
final_data['DLThQci7'] = final_data['L.Thrp.bits.DL.QCI.7(bit)']
final_data['DLThQci7Max'] = final_data['L.Thrp.bits.DL.QCI.7.Max(bit)']
final_data['DLThQci8'] = final_data['L.Thrp.bits.DL.QCI.8(bit)']
final_data['DLThQci8Max'] = final_data['L.Thrp.bits.DL.QCI.8.Max(bit)']
final_data['DLThQci9'] = final_data['L.Thrp.bits.DL.QCI.9(bit)']
final_data['DLThQci9Max'] = final_data['L.Thrp.bits.DL.QCI.9.Max(bit)']

final_data['ULThQci'] = final_data['L.Thrp.bits.UL(bit)']
final_data['ULThQciMax'] = final_data['L.Thrp.bits.UL.Max(bit)']
final_data['ULThQci5'] = final_data['L.Thrp.bits.UL.QCI.5(bit)']
final_data['ULThQci5Max'] = final_data['L.Thrp.bits.UL.QCI.5.Max(bit)']
final_data['ULThQci6'] = final_data['L.Thrp.bits.UL.QCI.6(bit)']
final_data['ULThQci6Max'] = final_data['L.Thrp.bits.UL.QCI.6.Max(bit)']
final_data['ULThQci7'] = final_data['L.Thrp.bits.UL.QCI.7(bit)']
final_data['ULThQci7Max'] = final_data['L.Thrp.bits.UL.QCI.7.Max(bit)']
final_data['ULThQci8'] = final_data['L.Thrp.bits.UL.QCI.8(bit)']
final_data['ULThQci8Max'] = final_data['L.Thrp.bits.UL.QCI.8.Max(bit)']
final_data['ULThQci9'] = final_data['L.Thrp.bits.UL.QCI.9(bit)']
final_data['ULThQci9Max'] = final_data['L.Thrp.bits.UL.QCI.9.Max(bit)']

final_data.drop(['L.Thrp.bits.DL(bit)','L.Thrp.bits.DL.Max(bit)',
             'L.Thrp.bits.DL.QCI.5(bit)','L.Thrp.bits.DL.QCI.5.Max(bit)',
             'L.Thrp.bits.DL.QCI.6(bit)','L.Thrp.bits.DL.QCI.6.Max(bit)',
             'L.Thrp.bits.DL.QCI.7(bit)','L.Thrp.bits.DL.QCI.7.Max(bit)',
             'L.Thrp.bits.DL.QCI.8(bit)','L.Thrp.bits.DL.QCI.8.Max(bit)',
             'L.Thrp.bits.DL.QCI.9(bit)','L.Thrp.bits.DL.QCI.9.Max(bit)',             
             'L.Thrp.bits.UL(bit)','L.Thrp.bits.UL.Max(bit)',
             'L.Thrp.bits.UL.QCI.5(bit)','L.Thrp.bits.UL.QCI.5.Max(bit)',              
             'L.Thrp.bits.UL.QCI.6(bit)','L.Thrp.bits.UL.QCI.6.Max(bit)',             
             'L.Thrp.bits.UL.QCI.7(bit)','L.Thrp.bits.UL.QCI.7.Max(bit)',
             'L.Thrp.bits.UL.QCI.8(bit)','L.Thrp.bits.UL.QCI.8.Max(bit)',
             'L.Thrp.bits.UL.QCI.9(bit)','L.Thrp.bits.UL.QCI.9.Max(bit)'],axis=1,inplace=True)
#添加小时标签
final_data['hour']=final_data['Time'].apply(lambda x : x.hour)
#存储数据
final_data.to_csv("../output/Common_CellName_flow_data.csv",index=False)