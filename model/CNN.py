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
            nn.Conv2d(1, 32, 5, 1, 2),  ## 此处输出通道的选择
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.FC1 = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(2 * 5 * 5, 25),
            # nn.ReLU(),
        )
        self.FC2 = nn.Linear(64*7*7, class_num)

    def forward(self, input):
        tmp1 = self.c1(input)
        tmp2 = self.c2(tmp1)
        # tmp3 = self.FC1(tmp2)
        tmp4 = self.FC2(tmp2.reshape(input.shape[0], -1))
        return tmp4

    def loss_function(self, y_pred, y_true):
        return nn.CrossEntropyLoss()(y_pred, y_true)

    def metrics_function(self, y_pred, y_true):
        import numpy as np
        y_true = np.argmax(y_true.detach().numpy(), axis=1)
        y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
        return metrics.accuracy_score(y_true, y_pred)

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024
    print('模型总大小为：{:.3f}kB'.format(all_size))
    print('模型总大小为：{:}B'.format(param_size + buffer_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


if __name__ == '__main__':
    model = CNN(10)
    getModelSize(model)

