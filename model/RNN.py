"""
   Author: hxt
   Created: 2024/3/13 
"""
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable

from model.BaseModel import BaseModel


class RNN(BaseModel):

    def __init__(self, class_num: int):
        super().__init__()
        self.rnn = RNNCore(28*28, 128, 2)
        # self.rnn = nn.RNN(28*28, 128, 2, batch_first=True)
        self.FC1 = nn.Linear(128, class_num)

    def forward(self, input):
        input = input.reshape(-1, 28 * 28)  ## batch,feature
        out, hidden = self.rnn(input)
        # tmp1 = out[:, -1, :]  ## 这里是以二维图片输入训练的方式
        tmp1 = out[:,  :]
        tmp2 = self.FC1(tmp1)
        return tmp2

    def loss_function(self, y_pred, y_true):
        return nn.CrossEntropyLoss()(y_pred, y_true)

    def metrics_function(self, y_true, y_pred):
        import numpy as np
        y_true = np.argmax(y_true.detach().numpy(), axis=1)
        y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
        return metrics.accuracy_score(y_true, y_pred)


class RNNCore(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        '''
        a(t)=b1+【W】h(t-1)+【U】x(t),
        h(t)=tanh(a(t)),
        o(t)=b2+【V】h(t),
        y(t)=softmax(o(t)),
        '''
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)

        self.num_layers = num_layers  ## 几层RNN
        ## 第二层往上的训练参数
        self.U2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs, hidden=None):
        batch_len = inputs.shape[0]
        if hidden is None:
            hidden = Variable(torch.zeros(batch_len, self.hidden_size))

        a = Variable(torch.zeros(batch_len, self.hidden_size))
        o = Variable(torch.zeros(batch_len, self.hidden_size))
        y = Variable(torch.zeros(batch_len, self.hidden_size))

        y = inputs
        for layer in range(self.num_layers):
            if layer == 0:
                a = self.W(hidden) + self.U(y)  # 1024, 28*28
            else:
                a = self.W(hidden) + self.U2(y)  # 1024, 28*28
            hidden = self.tanh(a)
            o = self.V(hidden)
            y = self.softmax(o)
        return y, hidden

