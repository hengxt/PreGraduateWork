"""
   Author: hxt
   Created: 2024/3/13 
"""
from abc import abstractmethod

import torch.nn as nn

from model.AE import AE
from model.CNN import CNN
from model.LSTM import LSTM
from model.RNN import RNN
import logging

logger = logging.getLogger(__name__)


class ModelFactory():

    def __init__(self):
        pass

    @staticmethod
    def getModel(modelName: str, **kwargs):
        if modelName == 'cnn':
            class_num = kwargs['class_num']
            return CNN(class_num)
        elif modelName == 'lstm':
            class_num = kwargs['class_num']
            return LSTM(class_num)
        elif modelName == 'rnn':
            class_num = kwargs['class_num']
            return RNN(class_num)
        elif modelName == 'AE':
            class_num = kwargs['class_num']
            return AE(class_num)
        else:
            logger.error('ModelFactory received an unknown Model %s', modelName)
            raise Exception(f"Unrecognized Model: {modelName}")
