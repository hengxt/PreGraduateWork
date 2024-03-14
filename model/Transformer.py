"""
   Author: hxt
   Created: 2024/3/14
   ## TODO dev
"""
import torch.nn as nn

d_model = 256

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self):
        pass

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiHeadAttention = MultiHeadAttention()
        self.FC1 = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input):
        tmp1 = self.multiHeadAttention(input)
        tmp2 = self.FC1(tmp1)
        tmp3 = self.norm(tmp1 + tmp2)
        return tmp3



class EncoderLayer(nn.Module):

    def __init__(self, module_num):
        super().__init__()
        self.encoders = nn.ModuleList([Encoder() for _ in range(module_num)])

    def forward(self, input):
        tmp1 = input
        for encoder in self.encoders:
            tmp1 = encoder(tmp1)
        return tmp1

class Transformer(nn.Module):

    def __init__(self, class_num):
        super().__init__()
        self.encoderLayer = EncoderLayer(6)
        self.decoderLayer = DecoderLayer(6)
        self.FC1 = nn.Sequential(
            nn.Linear(d_model, class_num, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_src, input_target):
        tmp1 = self.encoderLayer(input)
        tmp2 = self.decoderLayer(tmp1, input_target)
        tmp3 = self.FC1(tmp2)
        return tmp3
