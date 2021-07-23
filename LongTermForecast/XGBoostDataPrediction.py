import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from ShortTermForecast.DataPreProcess import *

def XGBoost_Predict(data, predictTime):
    # 设置x
    x1 = []
    # 设置y
    y1 = []
    # 设置seq_len
    seq_len = 30
    # 预测出来的维度，既一次性会预测出7个数据，利用30个数预测7个数
    prdict_size = 1

    # 确定训练集的窗口长度 用t-2, t-1,t次的消费间隔进行模型训练 用t+1次间隔对结果进行验证。
    for i in range(len(data) - seq_len):
        x1.append(data.iloc[i: i + seq_len])  # 分成seq_len个seq_len个一组
        y1.append(data.iloc[i + seq_len])  # 取第seq_len+1个数到最后一个数的值
    # 训练集
    train_all_x = torch.tensor(x1[:2000])
    train_all_y = torch.tensor(y1[:2000])
    # 测试集
    test_all_x = torch.tensor(x1[2000:])
    test_all_y = torch.tensor(y1[2000:])

    # learning_rate:学习率，n_estimators:弱分类器数目，max_depth:树的最大深度，min_child_weight:决定最小叶子节点样本权重和，gamma:节点分裂所需的最小损失函数下降值，
    # subsample:控制对于每棵树，随机采样的比例，olsample_bytree:用来控制每棵随机采样的列数的占比，seed:随机数的种子
    XGBoost = XGBRegressor(learning_rate=0.2,
                           max_depth=5,
                           n_estimators=500,
                           min_child_weight=1,
                           gamma=0,
                           subsample=0.8,
                           colsample_bytree=0.8,
                           seed=27
                           )

    # 交叉验证
    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(train_all_x):
        # 训练集, train_x:[x,seq_len,1],train_y:[y,prdict_size]
        train_x = train_all_x[train_index].numpy()
        train_y = train_all_y[train_index].numpy()
        # 测试集
        test_x = train_all_x[train_index].numpy()
        test_y = train_all_y[train_index].numpy()
        # 调用xgboost跑数据
        XGBoost.fit(train_x, train_y, early_stopping_rounds=50, eval_set=[(test_x, test_y)], verbose=True)

    # 测试
    prediction = []
    predictionTestData = XGBoost.predict(test_all_x.numpy())
    predictionTrainData = XGBoost.predict(train_all_x.numpy())
    prediction.extend(predictionTrainData)
    prediction.extend(predictionTestData)
    test_loss = mean_squared_error(predictionTestData, test_all_y)
    print("test_loss:", test_loss)

    # 未来预测
    # 未来数据
    futurePredictionData = []
    # 获得原始数据的倒数seq_len个数据，因为是根据seq_len个数据预测后面7天数据
    futurePredictionOriginalData = data.iloc[-seq_len:].values.reshape(1,-1)
    # 遍历划分获得输入的数据
    for i in range(predictTime - prdict_size + 1):
        # 用模型预测数据
        futurePrediction = XGBoost.predict(futurePredictionOriginalData)
        # 把第一个数加入到未来数据
        futurePredictionData.append(futurePrediction)
        # 把预测出来第一个数加入到输入数据中，作为输入来预测后面的1天
        futurePredictionOriginalData = np.insert(futurePredictionOriginalData, -1, futurePrediction, axis=1)
        # 获得原始数据的倒数seq_len个数据，因为前面会加入一个新的预测数据作为输入数据
        futurePredictionOriginalData = futurePredictionOriginalData[:, -seq_len:]
    # 画出预测的未来数据
    plt.plot(np.arange(len(data), len(data) + predictTime), futurePredictionData, label="LSTMFutureData")

    # 画图
    plt.plot(np.arange(2400, len(data)), data[2400:len(data)], label='TrueData')
    plt.plot(np.arange(2400, len(data)), prediction[2400 - seq_len:len(prediction)], label='LSTMFittingData')
    plt.ylabel("Temperature")
    plt.xlabel("Date")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

# compressionData = DataPreprocessing("2020年数据.xlsx", "温度监测")
# XGBoost_Predict(compressionData.iloc[:, 0], 30)