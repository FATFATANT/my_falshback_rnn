from enum import Enum

import torch
import torch.nn as nn


class Rnn(Enum):
    """ The available RNN units """

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):  # 返回名字对应的值
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        raise ValueError('{} not supported in --rnn'.format(name))


class RnnFactory:
    """ Creates the desired RNN unit. """

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):  # 按需返回torch中的模型
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)


class Flashback(nn.Module):
    """ Flashback RNN: Applies weighted average using spatial and temporal data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    """

    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory):
        super().__init__()
        self.input_size = input_size  # 总地点数
        self.user_count = user_count  # 总用户数
        self.hidden_size = hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight

        self.encoder = nn.Embedding(input_size, hidden_size)  # location embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size)  # user embedding
        self.rnn = rnn_factory.create(hidden_size)
        # create outputs in length of locations  2*hidden_size应是由于用户和地点embedding同时会作为输入，inputs_size应是输出这些个地点的概率
        self.fc = nn.Linear(2 * hidden_size, input_size)

    def forward(self, x, t, s, y_t, y_s, h, active_user):
        seq_len, user_len = x.size()  # 序列长为20，用户总数为200
        x_emb = self.encoder(x)
        out, h = self.rnn(x_emb, h)  # out是seq中每一步的输出，h是更新了20次后的h

        # compute weights per user
        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)  # (20, 200, 10)
        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)  # (200, 1)
            for j in range(i + 1):  # +1是因为range左闭右开
                dist_t = t[i] - t[j]  # 长度为200意味着同时计算200个用户当前时刻与前一时刻的时隙
                dist_s = torch.norm(s[i] - s[j], dim=-1)  # todo 这个求范数应是相当于经纬度相减后求直线距离，单位仍为度
                a_j = self.f_t(dist_t, user_len)  # todo 代码中这个user_len似乎传进去也没什么用
                b_j = self.f_s(dist_s, user_len)
                a_j = a_j.unsqueeze(1)  # (200, 1)
                b_j = b_j.unsqueeze(1)  # (200, 1)
                w_j = a_j * b_j + 1e-10  # small epsilon to avoid 0 division
                sum_w += w_j  # 权重累加用于最后归一化
                out_w[i] += w_j * out[j]  # 给RNN在序列中每一步的输出转变为从当前步回溯到第一步的各个输出结果的加权和
            # normalize according to weights
            out_w[i] /= sum_w  # 两个tensor相除是按元素除，这个sum_w依次计算200个用户的1步回溯、2步回溯、3步...，有几步就把几步的权重累加最后一除归一化

        # add user embedding:
        p_u = self.user_encoder(active_user)  # (1, 200, 10)  对应的用户向量
        p_u = p_u.view(user_len, self.hidden_size)  # (200, 10)  去掉长度为1的batch_size
        out_pu = torch.zeros(seq_len, user_len, 2 * self.hidden_size, device=x.device)  # 相当于个空数组，用于组装地点embedding和用户embedding
        for i in range(seq_len):
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)  # todo 应是指每个序列都要加上一份同样的用户embedding
        y_linear = self.fc(out_pu)  # (20, 200, 106994)
        return y_linear, h


'''
~~~ h_0 strategies ~~~
Initialize RNNs hidden states
'''


def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)


class H0Strategy:

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    """ use fixed normal noise as initialization """

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):  # 一个初始化h0复制batch_size份，即200个长度为10的tensor
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)  # (1, 200, 10)

    def on_reset(self, user):
        return self.h0  # (10)


class LstmStrategy(H0Strategy):
    """ creates h0 and c0 using the inner strategy """

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):  # 初始隐状态
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return h, c

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return h, c
