"""
   Author: hxt
   Created: 2024/3/13 
"""
import numpy as np
from torch.utils.data import Dataset
import torch


class MNIST(Dataset):
    def __init__(self, train: bool = True):
        with np.load('data/MNIST/mnist.npz', allow_pickle=True) as f:
            (x_train, y_train) = (f['x_train'], f['y_train'])
            (x_test, y_test) = (f['x_test'], f['y_test'])
        self.x_train = torch.Tensor(x_train.reshape(-1, 1, 28, 28) / 255)  # 通道
        self.x_test = torch.Tensor(x_test.reshape(-1, 1, 28, 28) / 255)  # 通道
        self.y_train = torch.Tensor(np.eye(10)[y_train])
        self.y_test = torch.Tensor(np.eye(10)[y_test])
        self.train = train

    def getValData(self):
        return self.x_test, self.y_test

    def __getitem__(self, index):
        if self.train:
            return self.x_train[index], self.y_train[index]
        else:
            return self.x_test[index], self.y_test[index]

    def __len__(self):
        if self.train:
            return self.x_train.shape[0]
        else:
            return self.x_test.shape[0]



'''
def MNIST(train: bool = True):
    with np.load('mnist.npz', allow_pickle=True) as f:
        (x_train, y_train) = (f['x_train'], f['y_train'])
        (x_test, y_test) = (f['x_test'], f['y_test'])
        x_train = x_train.reshape(-1, 1, 28, 28) / 255  # 通道
        x_test = x_test.reshape(-1, 1, 28, 28) / 255  # 通道
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]
        if train:
            return x_train, y_train
        else:
            return x_test, y_test
'''
