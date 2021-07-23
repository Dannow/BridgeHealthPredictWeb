import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
from sklearn.cluster import DBSCAN

# 有两个检测数据的表名
sheetNameDoubleData = ["应力监测", "温度监测", "索力监测"]
# 单个监测数据的表名
sheetNameSingleData = ["伸缩缝监测", "沉降监测"]


# 归一化函数
def Normalized(interpolatedData):
    return (interpolatedData - interpolatedData.min()) / (interpolatedData.max() - interpolatedData.min())


# 插入缺失值
def MissingValueProcessing(dataPath, sheetName):
    # 读取数据
    originalData = pd.read_excel(dataPath, sheet_name=sheetName, index_col="Unnamed: 0")
    # 获得缺失值所在的行
    nullData = originalData.loc[originalData.isnull().any(axis=1)]
    # 获得缺失值所在的列
    nullDataColumn = originalData.columns[originalData.isnull().any(axis=0)]
    # 插值后的数据
    interpolatedData = originalData
    # 判断是否有缺失值。如果有差值，使用同类均值插补
    if nullData.empty == False:
        # 判断该表是否有两列数据。是：用前一列来补；不是：用同一时间的分组的均值来补
        if sheetName in sheetNameDoubleData:
            # 遍历所有缺失值的列
            for i in range(nullDataColumn.size):
                # 求同个位置不同传感器的差值平均值
                mean = (originalData.loc[:, nullDataColumn[i][0] + "1"] - originalData.loc[:, nullDataColumn[i][0] + "2"]).mean()
                # 判断缺失值所在列的列名后面是奇数还是偶数
                if int(nullDataColumn[i][1:]) % 2 == 0:
                    # 使用同类均值插补（前一列+均值）
                    interpolatedData.fillna(value={nullDataColumn[i]: originalData[nullDataColumn[i][0] + "1"]+mean}, inplace =True)
                else:
                    # 使用同类均值插补（后一列+均值）
                    interpolatedData.fillna(value={nullDataColumn[i]: originalData[nullDataColumn[i][0] + "2"]+mean}, inplace=True)
        else:
            # 新增小时列，把索引的时间的小时赋值给新列
            interpolatedData["Hour"] = interpolatedData.index.str[11:-6]
            # 遍历所有缺失值的列
            for i in range(nullDataColumn.size):
                # 用按小时分组后的均值来进行插值
                interpolatedData[nullDataColumn[i]] = interpolatedData.groupby("Hour").transform(lambda x: x.fillna(x.mean()))
            # 删除新增的小时列
            interpolatedData.drop("Hour", axis=1, inplace=True)
    return interpolatedData


# 正常范围计算
def value_calculation(data, raw):
    mean = data.iloc[raw, 1:].mean()
    std = data.iloc[raw, 1:].std()
    upper = mean + 1.96 * std / math.sqrt(data.shape[1])
    lower = mean - 1.96 * std / math.sqrt(data.shape[1])
    if upper > 0 and lower > 0:
        upper = upper * 1.1  # 误差范围修正，可以根据需要适当扩大或缩小上下限
        lower = lower * 0.9
    elif upper > 0 and lower < 0:
        upper = upper * 1.1  # 误差范围修正，可以根据需要适当扩大或缩小上下限
        lower = lower * 1.1
    elif upper < 0 and lower < 0:
        upper = upper * 0.9  # 误差范围修正，可以根据需要适当扩大或缩小上下限
        lower = lower * 1.1
    elif upper == 0 or lower == 0:
        if upper == 0:
            upper = 0.1
        if lower == 0:
            lower = -0.1
    return lower, upper


# 异常值筛查
def OutlierElimination(data):
    # 统计异常值个数
    excepNumber = 0
    # 遍历每行找出异常值
    for i in range(data.shape[0]):
        # 获得最小值和最大值
        lower, upper = value_calculation(data, i)
        # 获取所有列
        tempdata = data.iloc[i, 0:]
        # 判断当前行所有列是否有超过最大或最小值的
        upValue = tempdata[tempdata >= upper].index.tolist()
        lowValue = tempdata[tempdata <= lower].index.tolist()
        # 把相应位置改成最大或最小值
        data.loc[data.index[i], upValue] = upper
        data.loc[data.index[i], lowValue] = lower
        # 统计异常值个数
        if (len(upValue) + len(upValue)) > 0:
            excepNumber += 1
    return data


# # 异常值剔除
# def OutlierElimination(interpolatedData):
#     # 对插值后的数据进行归一化
#     interpolateNormalizedData = Normalized(interpolatedData)
#     # 调用DBSCAN
#     db = DBSCAN(eps=0.25, min_samples=5).fit(interpolateNormalizedData)
#     # 获得聚类后的聚类结果（-1为离群点）
#     labels = db.labels_
#     # 新增一列，保存聚类结果
#     interpolatedData['cluster_db'] = labels
#     # print(interpolatedData.groupby("cluster_db").size())
#     # 输出异常值所在行
#     outliers = interpolatedData.loc[interpolatedData["cluster_db"] == -1].index
#     # print(interpolatedData.loc[interpolatedData["cluster_db"] == -1])
#     # 异常值用前一列代替
#     for i in outliers:
#         interpolatedData.loc[i] = interpolatedData.iloc[interpolatedData.index.get_loc(i)-1]
#     # # 删除异常值（离群点）
#     # interpolatedData.drop(outliers, inplace=True)
#     # 删除新增的聚类结果列cluster_db
#     interpolatedData.drop("cluster_db", axis=1, inplace=True)
#     return interpolatedData


# 数据压缩
def DataCompression(outlierProcessingData, sheetName):
    # 判断该表是否有两个传感器数据的表
    if sheetName in sheetNameDoubleData:
        # 获得列的大小
        columnSize = outlierProcessingData.columns.size
        # 遍历所有的奇数列，获得两列的均值赋值给新列
        for i in range(0, columnSize, 2):
            # 新列的列名
            columnName = outlierProcessingData.columns[i][0]+"_compression"
            # 把两列均值赋值给新列
            outlierProcessingData[columnName] = (outlierProcessingData[outlierProcessingData.columns[i]] + outlierProcessingData[outlierProcessingData.columns[i+1]])/2
        # 获得未新增列时的列个数（1,2，。。。）
        originalColumns = [j for j in range(columnSize)]
        # 删除原始的数据列
        outlierProcessingData.drop(outlierProcessingData.columns[originalColumns], axis=1, inplace=True)
    return outlierProcessingData

# 数据预处理
def DataPreprocessing(dataPath, sheetName):
    # 缺失值处理
    interpolatedData = MissingValueProcessing(dataPath, sheetName)
    # 异常值处理
    outlierProcessingData = OutlierElimination(interpolatedData)
    # 数据压缩
    compressionData = DataCompression(outlierProcessingData, sheetName)
    # 对压缩数据进行归一化
    compressionNormalizedData = Normalized(compressionData)
    print(compressionNormalizedData)
    return compressionNormalizedData, compressionData
