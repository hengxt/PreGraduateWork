"""
   Author: hxt
   Created: 2024/3/13 
"""
import torch
import torch.nn as nn
from sklearn import metrics

from model.BaseModel import BaseModel


class Encoder(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.FC1 = nn.Sequential(
            nn.Linear(in_size, in_size // 2),
            nn.ReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(in_size // 2, out_size),
            nn.ReLU()
        )

    def forward(self, input):
        tmp1 = self.FC1(input)
        tmp2 = self.FC2(tmp1)
        return tmp2


class Decoder(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.FC1 = nn.Sequential(
            nn.Linear(in_size, in_size * 2),
            nn.ReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(in_size * 2, out_size),
            nn.ReLU()
        )

    def forward(self, input):
        tmp1 = self.FC1(input)
        tmp2 = self.FC2(tmp1)
        return tmp2


class AE(BaseModel):

    def __init__(self, class_num):
        super().__init__()
        self.encoder = Encoder(28*28, 10)
        self.decoder = Decoder(10, 28*28)
        self.FC1 = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

    def forward(self, input):
        input = input.reshape(-1, 28*28)
        tmp1 = self.encoder(input)
        tmp2 = self.decoder(tmp1)
        return self.FC1(tmp2)
        # return [tmp1, tmp2, input]

    def loss_function(self, y_pred, y_true):
        return nn.CrossEntropyLoss()(y_pred, y_true)

    def metrics_function(self, y_pred, y_true):
        import numpy as np
        y_true = np.argmax(y_true.detach().numpy(), axis=1)
        y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
        return metrics.accuracy_score(y_true, y_pred)

'''
    def loss_function(self, y_pred, y_true):
        ## y_pred是forward函数的结果
        [tmp1, tmp2, input] = y_pred
        loss_data = torch.mean((tmp2 - input) ** 2)
        return loss_data

    def metrics_function(self, y_pred, y_true):
        [tmp1, tmp2, input] = y_pred
        import numpy as np
        y_true = np.argmax(y_true.detach().numpy(), axis=1)
        y_pred = np.argmax(tmp1.detach().numpy(), axis=1)
        return metrics.accuracy_score(y_true, y_pred)
'''
