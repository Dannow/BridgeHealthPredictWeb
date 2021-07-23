from ShortTermForecast.LSTMDataPredict import *
from ShortTermForecast.DataPreProcess import *
from django.shortcuts import HttpResponse
from utils.RegexMap import RegexMap
from utils.Denormalization import Denormalization


# 短期预测主函数。sheetName：字表名，columnIndex：列索引号，predictTime：预测时间
def ShortTermForcastMain(request):
    # 表名
    sheetName = request.GET.get("sheetName")
    # 列索引号
    columnIndex = int(request.GET.get("columnIndex"))
    # 预测时间
    predictTime = int(request.GET.get("predictTime"))
    # 传感器名
    sensorName = request.GET.get("sensorName")

    # 传感器名和模型的映射关系
    modleDict = {"温度传感器A.*?": "./ShortTermForecast/Modle/Temperature/Temperature_A_LSTM.pth",
                 "温度传感器B.*?": "./ShortTermForecast/Modle/Temperature/Temperature_B_LSTM.pth",
                 "温度传感器C.*?": "./ShortTermForecast/Modle/Temperature/Temperature_C_LSTM.pth",
                 "温度传感器D.*?": "./ShortTermForecast/Modle/Temperature/Temperature_D_LSTM.pth",
                 "应力传感器A.*?": "./ShortTermForecast/Modle/Stress/Stress_A_LSTM.pth",
                 "应力传感器B.*?": "./ShortTermForecast/Modle/Stress/Stress_B_LSTM.pth",
                 "应力传感器C.*?": "./ShortTermForecast/Modle/Stress/Stress_C_LSTM.pth",
                 "应力传感器D.*?": "./ShortTermForecast/Modle/Stress/Stress_D_LSTM.pth",
                 "索力传感器A.*?": "./ShortTermForecast/Modle/CableForce/CableForce_A_LSTM.pth",
                 "索力传感器B.*?": "./ShortTermForecast/Modle/CableForce/CableForce_B_LSTM.pth",
                 "索力传感器C.*?": "./ShortTermForecast/Modle/CableForce/CableForce_C_LSTM.pth",
                 "索力传感器D.*?": "./ShortTermForecast/Modle/CableForce/CableForce_D_LSTM.pth",
                 }
    print()
    reg_fee_dic = RegexMap(modleDict, None)
    modleName = reg_fee_dic[sensorName]

    # 获得预处理后数据
    compressionNormalizedData, compressionData = DataPreprocessing("./ShortTermForecast/202005数据.xls", sheetName)
    data = compressionNormalizedData.iloc[:, columnIndex]
    originalData = compressionData.iloc[:, columnIndex]
    # 预测指定的天数
    futurePredictionData = LSTM_DataPredict(data, predictTime, modleName)
    # 对预测出来的数据反归一化
    futurePredictionDenormalizationData = Denormalization(originalData,futurePredictionData)
    # 将返回数据加入，分隔符
    prediction = []
    for i in range(len(futurePredictionDenormalizationData)):
        prediction.append(futurePredictionDenormalizationData[i])
        prediction.append(",")

    return HttpResponse(prediction)