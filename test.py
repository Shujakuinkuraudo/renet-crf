import torch
from pytorch_partial_crf import CRF, PartialCRF

# # Create
# num_tags = 6
# model = CRF(num_tags)
#
# batch_size, sequence_length = 3, 5
# emissions = torch.randn(batch_size, sequence_length, num_tags)
# print(emissions)
# tags = torch.LongTensor([
#     [1, 2, 3, 3, 5],
#     [1, 3, 4, 2, 1],
#     [1, 0, 2, 4, 4],
# ])
#
# # Computing negative log likelihood
#
# model(emissions, tags)
#
# model.viterbi_decode(emissions)
# possible_tags = torch.randn(batch_size, sequence_length, num_tags)
# possible_tags[possible_tags <= 0] = 0  # `0` express that can not pass.
# possible_tags[possible_tags > 0] = 1  # `1` express that can pass.
# possible_tags = possible_tags.byte()
# print(model.restricted_viterbi_decode(emissions, possible_tags))

import torch
import torch.nn as nn


# 定义双向 LSTM 模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, prop_tags):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, prop_tags)  # *2 是因为双向 LSTM

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # 前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 进行全连接层操作并调整输出形状
        out = self.fc(out)
        out = out.view(out.size(0), out.size(1), -1)
        return out


class BiLSTM_crf(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, prop_tags):
        super().__init__()
        self.bilstm = BiLSTM(input_size, hidden_size, num_layers, prop_tags)
        self.crf = PartialCRF(prop_tags)

    def forward(self, x, tags):  # x[batch_size,seq_len,input_size]
        emissions = self.bilstm(x)
        out = self.crf(emissions, tags)
        return out

    def bilstm_forward(self, x):
        import numpy as np
        return np.array(self.crf.viterbi_decode(self.bilstm(x)))

# if __name__ == "__main__":
#     # 定义输入数据的维度和超参数
#     input_size = 10  # 输入特征维度
#     hidden_size = 20  # LSTM 隐藏层维度
#     num_layers = 2  # LSTM 层数
#     prop_tags = 5  # 每个时间步的属性标签数
#
#     # 创建双向 LSTM 模型实例
#     model = BiLSTM(input_size, hidden_size, num_layers, prop_tags)
#
#     # 创建输入数据
#     batch_size = 3
#     sequence_length = 5
#     x = torch.randn(batch_size, sequence_length, input_size)
#
#     # 前向传播
#     outputs = model(x)
#     print(outputs.shape)  # 输出维度：[batch_size, seqlen, prop_tags]
