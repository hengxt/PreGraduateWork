"""
   Author: hxt
   Created: 2024/3/13 
"""
import torch.nn as nn
from sklearn import metrics

from model.BaseModel import BaseModel


class CNN(BaseModel):

    def __init__(self, class_num:int) -> None:
        super().__init__()
        # input (1, 28, 28)
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 3, 5, ),  ## 此处输出通道的选择
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(3, 2, 3, ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.FC1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 5 * 5, 25),
            nn.ReLU(),
        )
        self.FC2 = nn.Linear(25, class_num)

    def forward(self, input):
        tmp1 = self.c1(input)
        tmp2 = self.c2(tmp1)
        tmp3 = self.FC1(tmp2)
        tmp4 = self.FC2(tmp3)
        return tmp4

    def loss_function(self, y_pred, y_true):
        return nn.CrossEntropyLoss()(y_pred, y_true)

    def metrics_function(self, y_pred, y_true):
        import numpy as np
        y_true = np.argmax(y_true.detach().numpy(), axis=1)
        y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
        return metrics.accuracy_score(y_true, y_pred)