import torch
from torch import nn, optim
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import TimeSeriesSplit
from ShortTermForecast.DataPreProcess import *
from ShortTermForecast.LSTMNet import Net

def LSTM_DataTrain(data, predictTime):
    # 设置x
    x1 = []
    # 设置y
    y1 = []
    # 设置seq_len
    seq_len = 30

    # 确定训练集的窗口长度 用t-2, t-1,t次的消费间隔进行模型训练 用t+1次间隔对结果进行验证。
    for i in range(len(data) - seq_len):
        x1.append(data.iloc[i: i + seq_len])  # 分成seq_len个seq_len个一组
        y1.append(data.iloc[i + seq_len])  # 取第seq_len+1个数到最后一个数的值
    # 训练集
    train_all_x = torch.tensor(x1[:4000])
    train_all_y = torch.tensor(y1[:4000])
    # 测试集
    test_all_x = torch.tensor(x1[4000:]).float().view(-1, seq_len, 1)
    test_all_y = torch.tensor(y1[4000:]).float().view(-1, 1)

    # 学习率
    lr = 0.01

    # 创建模型对象
    LSTM = Net()
    # 定义MSE损失函数
    criterion = nn.MSELoss()
    # 定义梯度优化，lr:学习率
    optimizer = optim.Adam(LSTM.parameters(), lr)

    # 交叉验证
    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(train_all_x):
        # 训练集, train_x:[x,seq_len,1],train_y:[y,prdict_size]
        train_x = torch.tensor(train_all_x[train_index]).float().view(-1, seq_len, 1)
        train_y = torch.tensor(train_all_y[train_index]).float().view(-1, 1)
        # 测试集
        test_x = torch.tensor(train_all_x[train_index]).float().view(-1, seq_len, 1)
        test_y = torch.tensor(train_all_y[train_index]).float().view(-1, 1)

        # 训练
        LSTM.train()
        # 迭代模型次数
        for iter in range(181):
            # 调用自己编写的RNN模型，output:[sample_nums, output_size]
            output = LSTM(train_x)
            # 计算损坏函数值
            loss = criterion(output, train_y)
            # 将梯度初始化为0，既loss对w的偏导置为0
            LSTM.zero_grad()
            # 反向传播
            loss.backward()
            # 更新所有参数
            optimizer.step()

            # 每迭代60次输出一次训练集均方差和测试集均方差
            if iter % 60 == 0:
                # 每次用训练好模型放在测试集上跑，看看MSE
                pre = LSTM(test_x)
                # 测试集的损失函数值
                tess_loss = criterion(pre, test_y)
                # 分数（解释回归模型的方差得分）
                score = explained_variance_score(test_y.detach().numpy(), pre.detach().numpy())
                print("iter:{}, train_loss:{}, test_loss:{}, mse:{}".format(iter, loss.item(), tess_loss, score))

    # 测试
    LSTM.eval()
    # 测试时的预测数据
    prediction = []
    # 对测试集进行预测
    predictionTestData = LSTM(test_all_x)
    # 对训练集进行预测
    predictionTrainData = LSTM(torch.tensor(train_all_x).float().view(-1, seq_len, 1))
    # 结果添加到数组中
    prediction.extend(predictionTrainData.reshape(-1))
    prediction.extend(predictionTestData.reshape(-1))
    tess_loss = criterion(predictionTestData, test_all_y)
    print("test-Loss", tess_loss)

    # 保存训练的模型：
    torch.save(LSTM.state_dict(), "./Modle/Stress_A_LSTM.pth")

    # 未来预测
    # 未来数据
    futurePredictionData = []
    # 获得原始数据的倒数seq_len个数据，因为是根据seq_len个数据预测后面7天数据
    futurePredictionOriginalData = data.iloc[-seq_len:].values.tolist()
    # 遍历划分获得输入的数据
    for i in range(predictTime):
        # 用模型预测数据
        futurePrediction = list(
            LSTM(torch.tensor(futurePredictionOriginalData).float().view(-1, seq_len, 1)).reshape(-1))
        # 把第一个数加入到未来数据
        futurePredictionData.extend(futurePrediction)
        # 把预测出来第一个数加入到输入数据中，作为输入来预测后面的7天
        futurePredictionOriginalData.extend(futurePrediction)
        # 获得原始数据的倒数seq_len个数据，因为前面会加入一个新的预测数据作为输入数据
        futurePredictionOriginalData = futurePredictionOriginalData[-seq_len:]
    # 画出预测的未来数据
    plt.plot(np.arange(len(data), len(data) + predictTime), futurePredictionData, label="LSTMFutureData")

    # 画图
    plt.plot(np.arange(5500, len(data)), data[5500:len(data)], label='TrueData')
    plt.plot(np.arange(5500, len(data)), prediction[5500 - seq_len:len(prediction)], label='LSTMFittingData')
    plt.ylabel("Temperature")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# compressionData = DataPreprocessing("./ShortTermForecast/202005数据.xls", "应力监测")
# LSTM_DataTrain(compressionData.iloc[:, 0], 28)