"""
   Author: hxt
   Created: 2024/3/13 
"""
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable

from model.BaseModel import BaseModel


class LSTM(BaseModel):

    def __init__(self, class_num):
        super().__init__()
        # self.lstm = nn.LSTM(input_size=28*28,
        #                     hidden_size=128,
        #                     num_layers=2,
        #                     batch_first=True
        #                     )
        self.lstm = LSTMCore(input_size=28 * 28, hidden_size=128, num_layers=1)
        self.FC1 = nn.Linear(128, class_num)

    def forward(self, input):
        input = input.reshape(-1, 28 * 28)
        tmp1, (h_n, c_n) = self.lstm(input)
        # tmp2 = tmp1[:, -1, :]
        tmp2 = tmp1[:, :]
        tmp3 = self.FC1(tmp2)
        return tmp3

    def loss_function(self, y_pred, y_true):
        return nn.CrossEntropyLoss()(y_pred, y_true)

    def metrics_function(self, y_pred, y_true):
        import numpy as np
        y_true = np.argmax(y_true.detach().numpy(), axis=1)
        y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
        return metrics.accuracy_score(y_true, y_pred)


class LSTMCore(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        '''
        it =Sigmoid(Wi·[h(t-1),xt]+bi)
        Ct tanh(Wg.[ht-1,xt bc)
        f:=o(Ws.[ht-1,xt]+bf)
        C=f*Ct-1+it*Cr
        ot =o(Wo [hi-1,xt bo)
        ht ot tanh (Ci)
        '''
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.Wf = nn.Linear(input_size, hidden_size)  ## 遗忘门
        self.Wi = nn.Linear(input_size, hidden_size)  ## 输入门
        self.Wg = nn.Linear(input_size, hidden_size)  ## 真正的输入门
        self.Wo = nn.Linear(input_size, hidden_size)  ## 输出门

        self.Uf = nn.Linear(hidden_size, hidden_size)
        self.Ui = nn.Linear(hidden_size, hidden_size)
        self.Ug = nn.Linear(hidden_size, hidden_size)
        self.Uo = nn.Linear(hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.num_layers = num_layers  ## 几层RNN
        ## 第二层往上的训练参数

    def forward(self, inputs):
        batch_len = inputs.shape[0]

        ht = Variable(torch.zeros(batch_len, self.hidden_size))
        ct = Variable(torch.zeros(batch_len, self.hidden_size))
        for layer in range(self.num_layers):

            ft = self.sigmoid(torch.add(self.Wf(inputs), self.Uf(ht)))
            it = self.sigmoid(torch.add(self.Wi(inputs), self.Ui(ht)))
            ot = self.sigmoid(torch.add(self.Wo(inputs), self.Uo(ht)))
            gt = self.tanh(torch.add(self.Wg(inputs), self.Ug(ht)))
            ct = torch.add(torch.mul(ft, ct), torch.mul(it, gt))
            ht = torch.mul(ot, self.tanh(ct))

        return ht, (ct, None)
