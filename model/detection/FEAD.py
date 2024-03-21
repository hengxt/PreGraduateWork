"""
   Author: hxt
   Created: 2024/3/21 
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics

from model.BaseModel import BaseModel


class FeatureEncoder(BaseModel):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2), nn.ReLU(),
            nn.Linear(input_size // 2, hidden_size), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size // 2), nn.ReLU(),
            nn.Linear(input_size // 2, input_size), nn.ReLU(),
        )

    def forward(self, inputs):
        inputs_hidden = self.encoder(inputs)
        inputs_new = self.decoder(inputs_hidden)
        return inputs_new, inputs_hidden

    def loss_function(self, inputs_new, inputs):
        return nn.MSELoss()(inputs_new, inputs)

    def metrics_function(self, y_pred, y_true):
        pass


class AnomalyScoreGenerator(nn.Module):

    def __init__(self, inputs_size):
        super().__init__()
        Z_layer = [
            [inputs_size + 1, inputs_size // 2],
            [inputs_size // 2 + 1, inputs_size // 4],
            [inputs_size // 4 + 1, inputs_size // 8],
        ]
        self.Z = nn.ModuleList([
            nn.Sequential(
                nn.Linear(i[0], i[1]), nn.ReLU()
            )
            for i in Z_layer
        ])
        self.S0 = nn.Linear(inputs_size // 8, 1)

    def forward(self, r, e):
        zk = r
        for layer in self.Z:
            zk = torch.concat((e, zk), dim=1)
            zk = layer(zk)
        S0 = self.S0(zk)
        return S0


class FEAD(BaseModel):

    def __init__(self, inputs_size, FeatureEncoder, hidden_size):
        super().__init__()
        self.featureEncoder = FeatureEncoder
        self.scoreGenerator = AnomalyScoreGenerator(inputs_size + hidden_size)
        self.lambda_arg = 1
        self.a0 = 5

    def forward(self, inputs):
        inputs_new, inputs_hidden = self.featureEncoder(inputs)
        e = torch.norm(inputs_new - inputs, p=2, dim=1).reshape(-1, 1)  ## 求欧几里得距离
        r = (inputs_new - inputs) / e  ## r表征x
        ## inputs_hidden 即 hi, 从图上看是加入到了scoreGenerator的第一层
        inputs_new = torch.concat((r, inputs_hidden), dim=1)
        score = self.scoreGenerator(inputs_new, e)
        return score, e, r

    def loss_function(self, y_pred, y_true):
        score, e, r = y_pred
        a0 = self.a0
        score = score.reshape(-1)
        # e = torch.norm(r, p=2, dim=1)
        e = e.reshape(-1)
        loss_encoder = torch.sum(
            (1 - y_true) * e +
            y_true * torch.max(torch.zeros_like(e), a0 - e)
        )
        loss_d = torch.sum(
            (1 - y_true) * torch.abs(score) +
            y_true * torch.max(torch.zeros_like(e), a0 - score)
        )
        loss = loss_d + self.lambda_arg * loss_encoder
        return torch.sum(loss) / y_true.shape[0]

    def predict(self, x_test):
        y_test_pred,_,_ = self.forward(x_test)
        # tmp = torch.ones_like(y_test_pred) * self.a0
        # y_test_pred = tmp - y_test_pred
        return y_test_pred

    def metrics_function(self, y_pred, y_true):
        return FEAD.__aucPerformance(y_pred, y_true)

    @staticmethod
    def __aucPerformance(mse, labels):
        roc_auc = metrics.roc_auc_score(labels, mse)
        ap = metrics.average_precision_score(labels, mse)
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
        return roc_auc, ap

def getData(src, partRatio):
    f = pd.read_csv(src, header=None)
    x = f.iloc[:, 0:-1]
    x = torch.from_numpy(x.to_numpy()).float()
    y = f.iloc[:, -1]
    y = torch.from_numpy(y.to_numpy()).float()
    data_len = x.shape[0]
    feature = x.shape[1]
    ## 数据划分
    partNum = int(data_len * partRatio)
    x_test = x[partNum:]
    x_train = x[:partNum]
    y_test = y[partNum:]
    y_train = y[:partNum]
    return x_test, x_train, y_test, y_train, feature

if __name__ == '__main__':
    ## A.数据获取
    x_test, x_train, y_test, y_train, feature = getData(
        '../../data/dataset/nslkdd_normalization.csv',
        0.8
    )
    ## B.模型训练
    epochs = 50
    batchsize = 128
    device = 'cpu'
    learn_rate = 0.005
    hidden_size = feature // 4
    ## 1. ======== 训练AE ========
    print("## 1. 训练AE")
    model_ae = FeatureEncoder(feature, hidden_size)
    optimizer_function1 = torch.optim.Adam(model_ae.parameters(), lr=learn_rate)
    for epoch in range(epochs):
        x_train = x_train.to(device)
        optimizer_function1.zero_grad()
        inputs_new, _ = model_ae(x_train)
        loss_ae = model_ae.loss_function(inputs_new, x_train)
        print('Epoch: {}, loss_ae = {:.6f}'.format(epoch + 1, loss_ae))
        loss_ae.backward()
        optimizer_function1.step()

    ## 2. ======== 训练FEAD ========
    print("## 2.训练FEAD")
    model_fead = FEAD(feature, model_ae, hidden_size)
    # 冻结featureEncoder层的参数
    for name, param in model_fead.named_parameters():
        if "featureEncoder" in name:
            param.requires_grad = False
    optimizer_function2 = torch.optim.Adam(model_fead.parameters(), lr=0.005)
    for epoch in range(50):
        x_train = x_train.to(device)
        optimizer_function2.zero_grad()
        out = model_fead(x_train)
        loss_fead = model_fead.loss_function(out, y_train)
        print('Epoch: {}, loss_ae = {:.6f}'.format(epoch + 1, loss_fead))
        loss_fead.backward()
        optimizer_function2.step()

    ## 3. ======== 保存模型 ========
    print("## 3.存储模型中......")  ## 下面需要外部调用，不能占用主进程
    # torch.save(model_fead, '../../TrainedModel/FEAD_lr{}_batchsize{}.pt'.format(learn_rate, batchsize))

    ## 4. ======== 测试FEAD ========
    print("## 4.测试FEAD")
    model = model_fead
    # model = torch.load('../../TrainedModel/FEAD_lr{}_batchsize{}.pt'.format(learn_rate, batchsize))
    y_test_pred = model.predict(x_test)
    roc_auc, ap = model.metrics_function(
        y_test_pred.detach().numpy(), y_test.detach().numpy()
    )
