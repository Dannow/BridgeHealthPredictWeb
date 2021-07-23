# 开源包
import pandas as pd
import joblib
from LongTermForecast.dataPro import loadData


# 用于评估的数据长度 （java键入长度改变此变量）
length = 50
bridgeClass = 2 # 桥梁编号

RandomForestClassifier = joblib.load("./HealthValuePredict/Modle/0model.pkl")
GradientBoostingClassifier = joblib.load("./HealthValuePredict/Modle/1model.pkl")
LogisticRegression = joblib.load("./HealthValuePredict/Modle/2model.pkl")

stress, temper, expan, settle = loadData()
stress = stress.iloc[:,bridgeClass]
temper = temper.iloc[:,bridgeClass]
expan = expan.iloc[:,bridgeClass]
settle = settle.iloc[:,bridgeClass]
x = pd.concat([stress,temper,expan,settle],axis=1)
colName = ["stress","temper","expan","settle"]
x.columns = colName

pos1 = 0
pos2 = 0
pos3 = 0

def cheakPos(output):
    global pos1
    global pos2
    global pos3
    if output == 1:
        pos1 += 1
    elif output == 2:
        pos2 += 1
    elif output == 3:
        pos3 += 1

i = 1


for i in range(length):
    # data = x.iloc[i:i+1, :]
    data = x.iloc[len(x)-i-1:len(x)-i, :]
    output1 = RandomForestClassifier.predict(data)
    output2 = GradientBoostingClassifier.predict(data)
    output3 = LogisticRegression.predict(data)
    output = int((output1 + output2 + output3)/3)
    cheakPos(output)

def classOut():
    global pos1
    global pos2
    global pos3
    if pos1 / length >= 0.5:
        return 1
    elif pos2 / length >= 0.5:
        return 2
    elif pos3 / length >= 0.5:
        return 3
classOut()
