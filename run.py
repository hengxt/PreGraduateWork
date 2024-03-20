"""
   Author: hxt
   Created: 2024/3/13 
"""
import task
import logging
from log.log import set_log_config
logger = logging.getLogger(__name__)
set_log_config()
import random
random.seed(4337)

## CNN
CNN_kwargs = {
    'epoch': 10,
    'device': 'cpu',
    'batch_size': 128,
    'learning_rate': 0.001,
    'dataset': 'mnist',
    'model': 'cnn',
    'model_args': {
        'class_num': 10,
    }
}

## LSTM
lstm_kwargs = {
    'epoch': 5,
    'device': 'cpu',
    'batch_size': 1024,
    'learning_rate': 0.001,
    'dataset': 'mnist',
    'model': 'lstm',
    'model_args': {
        'class_num': 10,
    }
}

## RNN
rnn_kwargs = {
    'epoch': 5,
    'device': 'cpu',
    'batch_size': 4096,
    'learning_rate': 0.001,
    'dataset': 'mnist',
    'model': 'rnn',
    'model_args': {
        'class_num': 10,
        'hidden_size': 128
    }
}

## AE
ae_kwargs = {
    'epoch': 1,
    'device': 'cpu',
    'batch_size': 1024,
    'learning_rate': 0.001,
    'dataset': 'mnist',
    'model': 'AE',
    'model_args': {
        'class_num': 10,
    }
}

task.run(**CNN_kwargs)
# task.run(**rnn_kwargs)
# task.run(**lstm_kwargs)
# task.run(**ae_kwargs)

