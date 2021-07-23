# 反归一化
def Denormalization(originalData,predictData):
    predictDataDenormalization = []
    for i in range(len(predictData)):
        result = (predictData[i] * (originalData.max() - originalData.min())) + originalData.min()
        predictDataDenormalization.append(result)
    return predictDataDenormalization