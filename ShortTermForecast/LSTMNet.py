import torch
from torch import nn, optim

# 构建RNN网络结构
class Net(nn.Module):
    # 定义属性
    def __init__(self, ):
        # 数据维度
        self.input_size = 1
        # 隐藏层节点个数
        self.hidden_size = 18
        # 设置seq_len
        self.seq_len = 30
        # 输出节点个数，既输出层只有一个神经元
        self.output_size = 1
        super(Net, self).__init__()
        # 定义LSTM网络，batch_first=True：表示以batch_size作为第一个维度，既把原本的[seq_len, sample_nums(batch_size), data_vector] -> [sample_nums, seq_len, data_vector]，dropout=0.5:增强；鲁棒性
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, dropout=0.5, batch_first=True)
        # 最后一层，既隐藏层到输出层，输入：[x(sample_nums), hidden_size * seq_len] -> 输出：[x(sample_nums),output_size]
        self.linear = nn.Linear(self.hidden_size * self.seq_len, self.output_size)
        # 用于防止过拟合，一般加在全连接层
        self.dropout = nn.Dropout(0.5)

    # 前向传播，x：需要预测的数据，hidden_prev：既h0，数据格式[batch_size, seq_len, data_vector],因为前面设置batch_first=True
    def forward(self, x):
        # 调用LSTM,output:[sample_nums, seq_len, hidden_size]
        output, (hidden, cell) = self.rnn(x)
        # 把out变成2维，-1既为根据hidden_size维度来自动调整，[sample_nums, seq_len, hidden_size] -> [sample_nums, hidden_size * seq_len]
        output = output.reshape(-1, self.hidden_size * self.seq_len)
        # 加入dropout防止过拟合
        output = self.dropout(output)
        # 调用前面定义的最后一层，把out放到输出层中，输入:[sample_nums, hidden_size * seq_len] -> 输出:[sample_nums, output_size]
        out = self.linear(output)
        # 返回数据
        return out
