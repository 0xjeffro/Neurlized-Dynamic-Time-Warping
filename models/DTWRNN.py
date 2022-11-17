import torch
from torch import nn
import numpy as np


class DTW_RNN(nn.Module):
    def __init__(self, shaplets, batch_size):
        super(DTW_RNN, self).__init__()
        self.M, self.L, self.D = shaplets.shape
        self.B = batch_size
        self.u = nn.Parameter(torch.ones(1, self.M).cuda())
        self.u_bias = nn.Parameter(torch.zeros(1, self.M).cuda())
        self.m = nn.Parameter(shaplets.cuda())  # [M, L, D]
        self.pad = nn.ConstantPad1d(padding=(1, 0), value=torch.inf)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.dis = torch.nn.PairwiseDistance(p=1.0, eps=1e-16, keepdim=False)

    def forward(self, inputs, hidden):
        x_t = inputs.unsqueeze(2).cuda()  # [B, D, 1]
        assert (self.B, self.D, 1) == x_t.shape

        s_prev_batch = hidden  # [B, M ,L]
        assert (self.B, self.M, self.L) == s_prev_batch.shape

        x_t = x_t.matmul(self.u)  # [B, D, M]
        x_t = x_t + self.u_bias  # [B, D, M]
        assert (self.B, self.D, self.M) == x_t.shape

        x_t = x_t.unsqueeze(1).tile(1, self.L, 1, 1)
        assert (self.B, self.L, self.D, self.M) == x_t.shape
        x_t_ = x_t.permute(0, 3, 1, 2)  # [B, M, L, D]
        assert (self.B, self.M, self.L, self.D) == x_t_.shape

        m_batch = self.m.unsqueeze(0).tile(self.B, 1, 1, 1)
        assert (self.B, self.M, self.L, self.D) == m_batch.shape

        s_prev_batch = self.pad(s_prev_batch)
        s_prev_batch_ = -self.pool(-s_prev_batch)  # [B, M, L]
        assert (self.B, self.M, self.L) == s_prev_batch_.shape
        x_t_ = x_t_.cuda()

        s_prev_batch_ = s_prev_batch_.cuda()
        s_t = self.dis(m_batch, x_t_) + s_prev_batch_  # [B, M, L]
        assert (self.B, self.M, self.L) == s_t.shape

        o = self.get_o()
        o_t = torch.matmul(s_t, o)
        return o_t, s_t

    def init_hidden(self, inf=np.inf):
        inf_mat = torch.ones(self.M, self.L - 1) * inf
        zero_mat = torch.zeros(self.M, 1)
        hidden = torch.cat((zero_mat, inf_mat), 1)  # [M, L]
        hidden = hidden.unsqueeze(0).tile(self.B, 1, 1)
        assert (self.B, self.M, self.L) == hidden.shape
        return hidden.cuda()

    def get_o(self):
        o = torch.zeros(self.L)
        o[-1] = 1
        return o.cuda()


class Logic(nn.Module):
    def __init__(self):
        super(Logic, self).__init__()
        var = self.init_variable()
        self.softmax = nn.Softmax(dim=1)
        self.one = torch.autograd.Variable(var, requires_grad=False)
        self.one = self.one.cuda()

    def forward(self, inputs):
        t = self.softmax(-inputs)
        t = torch.matmul(t, self.one)
        return t

    def init_variable(self, n_f=50, f_map=[(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]):
        t = torch.zeros((n_f, len(f_map)))
        for i in range(len(f_map)):
            t[f_map[i][0]: f_map[i][1], i] = 1
        return t.cuda()


class Model(nn.Module):
    def __init__(self, shaplets, batch_size):
        super(Model, self).__init__()
        self.model = DTW_RNN(shaplets, batch_size)
        self.N = Logic()

    def forward(self, inputs):
        inputs = inputs.cuda()
        s = self.model.init_hidden()

        B = inputs.shape[0]  # batch_size
        N = inputs.shape[1]  # n_step
        o = None
        for i in range(N):
            input_x = inputs[:, i, :]  # [B, dim]
            o, s = self.model(input_x, s)
        o_ = self.N(o)
        return o_
