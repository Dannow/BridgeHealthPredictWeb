from LongTermForecast.dataPro import *
from LongTermForecast.GRUDataPredict import *
from utils.RegexMap import RegexMap
from django.shortcuts import HttpResponse

def LongTermForcastMain(request):
    # 表名
    sheetName = request.GET.get("sheetName")
    # 列索引号
    columnName = request.GET.get("columnName")
    # 预测时间
    predictHours = int(request.GET.get("predictHours"))
    # 传感器名
    sensorName = request.GET.get("sensorName")

    # 传感器名和模型的映射关系
    modleDict = {"温度传感器A.*?": "./LongTermForecast/Modle/Temperature/temp_A.pth",
                 "温度传感器B.*?": "./LongTermForecast/Modle/Temperature/temp_B.pth",
                 "温度传感器C.*?": "./LongTermForecast/Modle/Temperature/temp_C.pth",
                 "温度传感器D.*?": "./LongTermForecast/Modle/Temperature/temp_D.pth",
                 "温度传感器E.*?": "./LongTermForecast/Modle/Temperature/temp_E.pth",
                 "温度传感器F.*?": "./LongTermForecast/Modle/Temperature/temp_F.pth",
                 "温度传感器G.*?": "./LongTermForecast/Modle/Temperature/temp_G.pth",
                 "温度传感器H.*?": "./LongTermForecast/Modle/Temperature/temp_H.pth",
                 "应力传感器A.*?": "./LongTermForecast/Modle/Stress/stress_A.pth",
                 "应力传感器B.*?": "./LongTermForecast/Modle/Stress/stress_B.pth",
                 "应力传感器C.*?": "./LongTermForecast/Modle/Stress/stress_C.pth",
                 "应力传感器D.*?": "./LongTermForecast/Modle/Stress/stress_D.pth",
                 "应力传感器E.*?": "./LongTermForecast/Modle/Stress/stress_E.pth",
                 "应力传感器F.*?": "./LongTermForecast/Modle/Stress/stress_F.pth",
                 "应力传感器G.*?": "./LongTermForecast/Modle/Stress/stress_G.pth",
                 "应力传感器H.*?": "./LongTermForecast/Modle/Stress/stress_H.pth",
                 "沉降传感器A.*?": "./LongTermForecast/Modle/Settlement/settlement_A.pth",
                 "沉降传感器B.*?": "./LongTermForecast/Modle/Settlement/settlement_B.pth",
                 "沉降传感器C.*?": "./LongTermForecast/Modle/Settlement/settlement_C.pth",
                 "沉降传感器D.*?": "./LongTermForecast/Modle/Settlement/settlement_D.pth",
                 "沉降传感器E.*?": "./LongTermForecast/Modle/Settlement/settlement_E.pth",
                 "沉降传感器F.*?": "./LongTermForecast/Modle/Settlement/settlement_F.pth",
                 "沉降传感器G.*?": "./LongTermForecast/Modle/Settlement/settlement_G.pth",
                 "沉降传感器H.*?": "./LongTermForecast/Modle/Settlement/settlement_H.pth",
                 "伸缩缝传感器A.*?": "./LongTermForecast/Modle/Expansion/expansion_A.pth",
                 "伸缩缝传感器B.*?": "./LongTermForecast/Modle/Expansion/expansion_B.pth",
                 "伸缩缝传感器C.*?": "./LongTermForecast/Modle/Expansion/expansion_C.pth",
                 "伸缩缝传感器D.*?": "./LongTermForecast/Modle/Expansion/expansion_D.pth",
                 "伸缩缝传感器E.*?": "./LongTermForecast/Modle/Expansion/expansion_E.pth",
                 "伸缩缝传感器F.*?": "./LongTermForecast/Modle/Expansion/expansion_F.pth",
                 "伸缩缝传感器G.*?": "./LongTermForecast/Modle/Expansion/expansion_G.pth",
                 "伸缩缝传感器H.*?": "./LongTermForecast/Modle/Expansion/expansion_H.pth"}
    reg_fee_dic = RegexMap(modleDict, None)
    modleName = reg_fee_dic[sensorName]

    # 数据
    original = loadData_All(columnName, sheetName)
    data, sc = dataProcess(original)

    # 预测指定的天数
    futurePredictionData = GRU_DataPredict(data, predictHours, modleName)
    futurePredictionDenormalizationData = sc.inverse_transform(np.array(futurePredictionData).reshape(-1, 1))
    # print("futurePredictionDenormalizationData",futurePredictionDenormalizationData)

    # 将返回数据加入，分隔符
    prediction = []
    for i in range(len(futurePredictionDenormalizationData)):
        prediction.append(futurePredictionDenormalizationData[i][0])
        prediction.append(",")

    return HttpResponse(prediction)
