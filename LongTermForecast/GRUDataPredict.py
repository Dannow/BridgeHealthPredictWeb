# 开源包
import torch
import matplotlib.pyplot as plt
import numpy as np
#自定义包
from LongTermForecast.CRUNet import GRU
from LongTermForecast.dataPro import dataProcess
from LongTermForecast.dataPro import loadData_All
from LongTermForecast.dataPro import createTrainAndTestBy15Days


def GRU_DataPredict(data, predictHours, modleName):
    # 预测设置
    # hours = 20 # 最少预测一小时
    seq_len = 7

    # 数据读取，若使用自定义数据处理则删除本部分。
    # data = dataProcess(loadData_All('A1'))
    x, y, test_x, test_y = createTrainAndTestBy15Days(data, seq_len)
    # 数据格式：只使用测试集最后7小时进行预测


    # 模型构造
    hidden_size = 4
    model = GRU(input_size=1,hidden_size = hidden_size,output_size = 1, seq_len=seq_len)
    # 模型载入
    # model.load_state_dict(torch.load('./Modle/Settlement/settlement_H.pth'))
    model.load_state_dict(torch.load(modleName))
    model.eval()
    prediction = model(test_x)
    prediction = prediction.detach().numpy()

    # 预测数据
    predata = []
    data = data.tolist()
    temp_list = data[-seq_len:] # old_data 是list

    # 此处用于测试数据融合，还不完善，这里是没有融合的
    for i in range(predictHours):
        temp_tensor = torch.as_tensor(temp_list).float().view(-1, seq_len, 1)
        out_tenser = model(temp_tensor)  # new_data 是tensor
        new_data = out_tenser.detach().numpy()
        temp_list.extend(new_data)
        predata.extend(new_data)
        olddata = data[-i]
        temp_list = temp_list[-seq_len:]
        if (i > seq_len):
            temp_list[-1] = (olddata * 0.6 + temp_list[-1] * 0.4)

    return predata