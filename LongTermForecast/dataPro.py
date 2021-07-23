# 开源包
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def loadData_All(colom, sheetName):
    original = pd.read_excel('./LongTermForecast/2020年数据.xlsx', sheet_name=sheetName)
    data = original.loc[:, colom]
    return data

def loadData():
    stress_testing_hourData = pd.read_excel('./LongTermForecast/2020年数据.xlsx', sheet_name='应力监测')
    temperature_hourData = pd.read_excel('./LongTermForecast/2020年数据.xlsx', sheet_name='温度监测')
    expansion_joint_monitoring_hourData = pd.read_excel('./LongTermForecast/2020年数据.xlsx', sheet_name='伸缩缝监测')
    settlement_monitoring_hourData = pd.read_excel('./LongTermForecast/2020年数据.xlsx', sheet_name='沉降监测')

    return stress_testing_hourData, temperature_hourData, expansion_joint_monitoring_hourData, settlement_monitoring_hourData

def dataProcess(data):
    sc = MinMaxScaler(feature_range=(0, 1))
    sc_dataSet = sc.fit_transform(np.array(data).reshape(-1, 1))
    data = pd.DataFrame(sc_dataSet,columns=['A']) # 单列测试

    return data.iloc[:,0], sc

def createTrainAndTestBy15Days(data, seq_len):
    split = int(len(data)*0.8)
    print(split)
    train_data = data[:split]
    test_data = data[split:]

    train_x, train_y = dataCreate(train_data,seq_len) # 按模式1生成训练集
    test_x, test_y = dataCreate(test_data,seq_len)

    train_x = torch.as_tensor(train_x).float().view(-1, seq_len, 1)
    train_y = torch.as_tensor(train_y).float().view(-1, 1)
    test_x = torch.as_tensor(test_x).float().view(-1, seq_len, 1)
    test_y = torch.as_tensor(test_y).float().view(-1, 1)

    return train_x, train_y, test_x, test_y

def dataCreate(data, seq_len, mode = 0):
    x = []
    y = []
    data = data.astype("float32")
    if mode == 0:
        for i in range(len(data) - seq_len):
            x.append(data.values[i: i + seq_len])  # 分成seq_len个seq_len个一组
            y.append(data.values[i + seq_len])  # 取第seq_len+1个数到最后一个数的值
    else:
        for i in range(len(data) - 50 * seq_len):
            x.append(data.values[i: i + seq_len])
            y.append(data.values[i + 50 * seq_len])
    return x, y