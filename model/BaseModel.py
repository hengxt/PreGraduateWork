"""
   Author: hxt
   Created: 2024/3/13 
"""
from abc import abstractmethod

from torch import nn


class BaseModel(nn.Module):

    @abstractmethod
    def loss_function(self, y_pred, y_true):
        pass

    @abstractmethod
    def metrics_function(self, y_pred, y_true):
        pass