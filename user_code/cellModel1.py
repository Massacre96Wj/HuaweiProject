
#导入所需的库函数
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import numpy as np
starttime = datetime.now()#程序运行的开始时间

#推后几个小时，设置预测后几个小时
HOUR_OFFSET = 1
#测试小区
test_all_cell = False
test_cell_id = set(['393805-1'])

#修改训练集起始时间，测试集起始时间，测试集结束时间
train_start = '2017-4-15'
test_start = '2017-6-15'
test_end = '2017-6-15 23:00:00'
#Ridge参数
ridge_alpha = 0.001

'''
读取数据部分
'''
data=pd.read_csv("../data/month5-data.csv",delimiter=',',encoding='gbk')
data.drop("INDEX",axis=1,inplace=True)
data = data[['CellName','Time','L.Traffic.User.Avg','L.Traffic.User.Max']]
data['userCount'] = data['L.Traffic.User.Avg'] #将列名重新命名
data['userMax'] = data['L.Traffic.User.Max']
data.drop(['L.Traffic.User.Avg','L.Traffic.User.Max'],axis=1,inplace=True) #去掉重复的咧
data['Time']=data['Time'].apply(lambda x: datetime.strptime(x,"%Y/%m/%d %H:%M")) #将时间形式改成特定的格式
data['hour']=data['Time'].apply(lambda x : x.hour)
temp1 = data.copy()

data=pd.read_csv("../data/month6-data-1.csv",delimiter=',',encoding='gbk')
data.drop("INDEX",axis=1,inplace=True)
data = data[['CellName','Time','L.Traffic.User.Avg','L.Traffic.User.Max']]
data['userCount'] = data['L.Traffic.User.Avg']
data['userMax'] = data['L.Traffic.User.Max']
data.drop(['L.Traffic.User.Avg','L.Traffic.User.Max'],axis=1,inplace=True)
data['Time']=data['Time'].apply(lambda x: datetime.strptime(x,"%Y/%m/%d %H:%M"))
data['hour']=data['Time'].apply(lambda x : x.hour)
temp2 = data.copy()

data=pd.read_csv("../data/month6-data-2.csv",delimiter=',',encoding='gbk',header=None)
data1=pd.read_csv("../data/month6-data-1.csv",delimiter=',',encoding='gbk')
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

data2 = pd.concat([data2,data3]) #合并6月份的数据
'''
提取连续时间的数据段
'''
new_data1 = data1[(data1.Time>='2017-04-07') & (data1.Time<'2017-05-09')]
new_data2 = data1[(data1.Time>='2017-05-15') & (data1.Time<'2017-05-24')]
new_data3 = data2[(data2.Time>='2017-06-08') & (data2.Time<'2017-06-17')]

cell_set = set(new_data1['CellName'])&set(new_data2['CellName'])&set(new_data3['CellName'])#小区编号的交集

data = pd.concat([new_data1,new_data2,new_data3])
data['flag'] = data['CellName'].apply(lambda x: 1 if x in cell_set else 0)#判断小区编号是否在公共集合中，在赋值为1不在为0
data = data[data['flag'] == 1]
data.drop('flag',axis=1,inplace=True)

#筛选测试的小区
if not test_all_cell:
    data = data[data['CellName'].isin(test_cell_id)]


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
        res=pd.merge(res,tmp,on=["CellName","hour"],how='outer')
    
    res=pd.merge(res,to_data,on=['CellName','hour'],how='outer')
    return res
#提取前几个小时的在线用户数特征，以及前几个小时的在线用户数的最大值特征
def get_before_hour(data,timeStr,HOUR_OFFSET):
    hour_begin=HOUR_OFFSET
    fea_col=['userCount','userMax']
    hour_index=[[hour_begin,hour_begin+1,hour_begin+2,hour_begin+3],[hour_begin,hour_begin+1]]
    result=pd.DataFrame()
    for fea_index in range(len(fea_col)):
        tmp=get_before_hour_1(data,timeStr,fea_col[fea_index],hour_index[fea_index])
        if np.shape(result)[0]==0:
            result=tmp
        else:
            result=pd.merge(result,tmp,on=['CellName','Time','hour'],how='outer')
    return result

#提取前一天的在线用户数特征	
def get_before_day(data,timeStr):
    timeStr_before=(datetime.strptime(timeStr,"%Y-%m-%d")-timedelta(1)).strftime("%Y-%m-%d")
    data_before=data[data.Time==timeStr_before].copy()
    data_before.rename(columns={'userCount':'before1d_DL'},inplace=True)#给提取的特征所在的列命名
    data_=data[data.Time==timeStr].copy()
    data_=data_[["CellName","hour","Time"]].copy()
    data_before=data_before[["CellName","hour",'before1d_DL']].copy()
    data_=pd.merge(data_,data_before,on=['CellName','hour'],how='outer')
    return data_


def deal_missing_data_sub(data,fea_name):
    """
    处理缺失值，根据对应小区对应不同时刻的userCount和userMax的中位值进行填充
    """
    data=data[['CellName','Time','hour',fea_name]].copy()
    CellName_id=set(data.CellName)
    date_hour_range=pd.date_range("4/14/2017 00:00",pd.to_datetime(test_start)-timedelta(hours=1),freq='H')

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
    """
    处理缺失值，根据对应小区对应不同时刻的userCount和userMax的中位值进行填充
    """
    fea_col=['userCount','userMax']
    result=pd.DataFrame()
    for fea in fea_col:
        if np.shape(result)[0]==0:
            result=deal_missing_data_sub(data,fea)
        else:
            result=pd.merge(result,deal_missing_data_sub(data,fea),on=['CellName','Time','hour'],how='outer')
    return result


date_used=pd.date_range(train_start,test_end)
date_used=[x.strftime("%Y-%m-%d") for x in date_used]

data.Time = pd.to_datetime(data.Time)
data['Time']=data['Time'].apply(lambda x: x.strftime("%Y-%m-%d"))
data=deal_missing_data(data) #处理缺失数据

data_ts=pd.DataFrame()
sp_date=set()
for date_index  in range(len(date_used)):
    if date_used[date_index] in sp_date:
        continue       
    res=get_before_hour(data,date_used[date_index],HOUR_OFFSET)#根据HOUR_OFFSET偏移量的设置，提取前几个小时的特征
    res2=get_before_day(data,date_used[date_index]) #提取前一天的特征
    res=pd.merge(res,res2,on=['CellName','Time','hour'],how='outer')
    if date_index==0:
        data_ts=res
    else:
        data_ts=pd.concat([data_ts,res],axis=0)

data1 = data_ts[(data_ts.Time>='2017-04-15') & (data_ts.Time<'2017-05-08')]
data2 = data_ts[(data_ts.Time>='2017-05-17') & (data_ts.Time<'2017-05-23')]
data3 = data_ts[(data_ts.Time>='2017-06-09') & (data_ts.Time<='2017-06-15')]
data_ts = pd.concat([data1,data2,data3])

data_ts.sort_values(by=['CellName','hour'],inplace=True) #根据小区和hour的值进行升序排序
cell_id=list(set(data.CellName)) 

data_ts['Time'] = pd.to_datetime(data_ts['Time'])
data_ts['Time'] = data_ts[['Time','hour']].apply(lambda x: x[0] + timedelta(hours=x[1]), axis=1)
model_dict={}

test_time_list=pd.date_range(test_start,test_end,freq='H')

'''
对不同时刻不同小区分别建立岭模型，进行预测，并将输出的结果放到predict_result中
'''
predict_result=pd.DataFrame()
for time_index in test_time_list:
    test_hour=time_index.hour
    print(" 测试时间 :",time_index)

    predict_val=np.zeros(len(cell_id))
    real_x=np.zeros(len(cell_id))

    data_ts_tmp=data_ts[data_ts.hour==test_hour].copy()
    train_data=data_ts_tmp[data_ts_tmp.Time<time_index].copy()
    test_data=data_ts_tmp[data_ts_tmp.Time==time_index].copy()
    
    for cell_id_index in range(len(cell_id)):
        #lr=LinearRegression()
        lr=Ridge(alpha=ridge_alpha)
        train_data_tmp=train_data[train_data.CellName==cell_id[cell_id_index]].copy()
        train_data_tmp.drop(['CellName','Time','hour'],axis=1,inplace=True)
        train_x=train_data_tmp.drop(['userCount','userMax'],axis=1)
        
        lr.fit(train_x,train_data_tmp['userCount'])
    
        test_data_tmp=test_data[test_data.CellName==cell_id[cell_id_index]].copy()
        test_x=test_data_tmp.drop(['CellName','Time','hour','userCount','userMax'],axis=1)   
        pre_y=lr.predict(test_x)
        real_x[cell_id_index]=test_data_tmp.userCount.values
        predict_val[cell_id_index]=max(pre_y,0)

    result=pd.DataFrame()

    result['CellName']=cell_id
    result['predict_val']=predict_val
    result['real']=real_x
    result['time']=time_index
    
    #求取绝对误差
    result['absolute error']=result['real']-result['predict_val']
    result['absolute error']=result['absolute error'].apply(lambda x :abs(x))
    predict_result=pd.concat([predict_result,result],axis=0)

#求取相对误差
result_no_0=predict_result[predict_result.real>0].copy()
result_no_0['relative error']=result_no_0['absolute error']/result_no_0['real']

#输出不同阈值情况下的误差的均值
print("误差门限为0 误差为 ：",np.mean(result_no_0.loc[result_no_0.real>0,'relative error']))
print("误差门限为1 误差为 ：",np.mean(result_no_0.loc[result_no_0.real>1,'relative error']))
print("误差门限为2 误差为 ：",np.mean(result_no_0.loc[result_no_0.real>2,'relative error']))
print("误差门限为3 误差为 ：",np.mean(result_no_0.loc[result_no_0.real>3,'relative error']))
endtime = datetime.now() #程序运行结束时间
print(endtime-starttime) #输出程序运行的时间 
result_no_0.to_csv("../data/userCount_build_model_cell.csv",index=False) #预测结果存放的路径以及命名格式
