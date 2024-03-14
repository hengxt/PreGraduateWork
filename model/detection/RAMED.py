"""
   Author: hxt
   Created: 2024/3/14 
"""

import torch.nn as nn

from model.BaseModel import BaseModel

class RecurrentAutoencoder(nn.Module):

    def __init__(self, num_encoder):
        super().__init__()

    def forward(self):
        pass


class MultiEnsembleDecoder(nn.Module):

    def __init__(self, num_decoder):
        super().__init__()

    def forward(self):
        pass



class RAMED(BaseModel):

    def __init__(self):
        super().__init__()
        self.recurrentAutoencoder = RecurrentAutoencoder(3)
        self.multiEnsembleDecoder = MultiEnsembleDecoder(3)

    def forward(self, input_time):
        h = self.encoder(input_time)

        pass

    def loss_function(self, y_pred, y_true):
        pass


    def metrics_function(self, y_pred, y_true):

        pass