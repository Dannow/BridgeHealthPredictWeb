from ShortTermForecast.LSTMDataTrain import *
from ShortTermForecast.LSTMNet import Net


# 预测数据
def LSTM_DataPredict(data, predictTime, modleName):
    seq_len = 30

    # 导入训练好的模型
    LSTM = Net()
    LSTM.load_state_dict(torch.load(modleName))

    # 测试模型
    LSTM.eval()
    # 未来数据
    futurePredictionData = []
    # 获得原始数据的倒数seq_len个数据，因为是根据seq_len个数据预测后面7天数据
    futurePredictionOriginalData = data.iloc[-seq_len:].values.tolist()
    # 遍历划分获得输入的数据
    for i in range(predictTime):
        # 用模型预测数据
        futurePrediction = LSTM(torch.tensor(futurePredictionOriginalData).float().view(-1, seq_len, 1)).reshape(-1)
        # 把第一个数加入到未来数据
        futurePredictionData.extend(futurePrediction.detach().numpy())
        # 把预测出来第一个数加入到输入数据中，作为输入来预测后面的7天
        futurePredictionOriginalData.extend(futurePrediction)
        # 获得原始数据的倒数seq_len个数据，因为前面会加入一个新的预测数据作为输入数据
        futurePredictionOriginalData = futurePredictionOriginalData[-seq_len:]

    # 数据展示，前者为真实数据，后者为预测数据
    # plt.plot(np.arange(0, 126), data[-126:], label='Data')
    # plt.plot(np.arange(126, 126 + len(futurePredictionData)), futurePredictionData[0:len(futurePredictionData)], label='OneDay')
    #
    # plt.ylabel("settlement_H")
    # plt.xlabel("Date")
    # plt.legend()
    # plt.show()
    return futurePredictionData