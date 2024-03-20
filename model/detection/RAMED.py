"""
   Author: hxt
   Created: 2024/3/14 
"""
import torch.nn
import torch.nn as nn
from torch.autograd import Variable

from model.BaseModel import BaseModel

class RecurrentAutoencoder(nn.Module):

    """
    RNN的隐藏层，可以改成LSTM的隐藏层
    hidden = tanh(hidden_weight*hidden_tensor + input_weight*input_tensor)
    hidden(e;e=1,2···num_encoders)
    hidden(E) = MLP(concat(hidden(e);e=.....))
    input: <batchsize, seq_size>  --> output: <batchsize, hidden_size>
    """

    def __init__(self, num_encoder, input_size, hidden_size):
        super().__init__()
        self.num_encoder = num_encoder
        self.hidden_size = hidden_size

        self.Wh = nn.Linear(hidden_size, hidden_size)  ## 隐藏层
        self.Wi = nn.Linear(input_size, hidden_size)  ## 输入层
        self.tanh = nn.Tanh()
        self.MLP = nn.Linear(hidden_size*num_encoder, hidden_size)

    def forward(self, inputs, ht=None):
        batch_len = inputs.shape[0]
        if ht is None:
            ht = Variable(torch.zeros(batch_len, self.hidden_size))
        outputs = None
        for layer in range(self.num_encoder):
            ht = self.tanh(self.Wh(ht), self.Wi(inputs))
            if outputs is None:
                outputs = ht
            else:
                outputs = torch.concat(outputs, ht)
        outputs = self.MLP(outputs)
        return outputs


class MultiEnsembleDecoder(nn.Module):

    def __init__(self, num_decoder, input_size, hidden_size, output_size):
        super().__init__()
        self.num_decoder = num_decoder

        self.Wh = nn.Linear(hidden_size, hidden_size)  ## 隐藏层
        self.Wi = nn.Linear(input_size, hidden_size)  ## 输入层
        self.tanh = nn.Tanh()
        self.Wo = nn.Linear(hidden_size, output_size)  ## 输出层
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        batch_len = inputs.shape[0]

        ht = Variable(torch.zeros(batch_len, self.hidden_size))
        output = None
        for layer in range(self.num_decoder):
            ht = self.tanh(self.Wh(ht), self.Wi(inputs))
            output = self.softmax(self.Wo(ht))

        return output



class RAMED(BaseModel):

    def __init__(self):
        super().__init__()
        self.recurrentAutoencoder = RecurrentAutoencoder(3)
        self.multiEnsembleDecoder = MultiEnsembleDecoder(3)

    def forward(self, inputs):
        h_e = self.encoder(inputs)
        output = self.multiEnsembleDecoder(h_e)
        return output, inputs

    def loss_function(self, y_pred, y_true):
        output, inputs = y_pred
        sum_batch = None
        mse = nn.MSELoss()(output, inputs)

        loss = 1 / batch_len * sum_batch
        pass


    def metrics_function(self, y_pred, y_true):

        pass