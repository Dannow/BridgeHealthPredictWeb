import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, output_size):
        super(GRU, self).__init__()
        self.use_cuda = True
        self.P = seq_len  # 输入窗口大小
        self.m = input_size  # 列数，变量数
        self.hidR = hidden_size
        self.GRU = nn.GRU(self.m, self.hidR)

        self.linear = nn.Linear(self.hidR, self.m, output_size)

    def forward(self, x):
        # x: [batch, window, n_val]
        #         batch_size = x.shape[0]
        #         x_flat = x.view(batch_size, -1)
        x1 = x.permute(1, 0, 2).contiguous()  # x1: [window, batch, n_val]
        _, h = self.GRU(x1)  # r: [1, batch, hidRNN]
        h = torch.squeeze(h, 0)  # r: [batch, hidRNN]
        res = self.linear(h)  # res: [batch, n_val]
        return res
