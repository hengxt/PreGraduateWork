"""
   Author: hxt
   Created: 2024/3/13 
"""
import logging

import numpy as np
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DataFactory
from model import ModelFactory
from util import ImgUtil

logger = logging.getLogger(__name__)


def run(**kwargs):
    ## parameter
    device = kwargs['device']
    epochs = kwargs['epoch']
    batch_size = kwargs['batch_size']
    learn_rate = kwargs['learning_rate']
    dataset = kwargs['dataset']
    model = kwargs['model']
    model_args = kwargs['model_args']
    ## get data
    logger.info('select dataset: {}'.format(dataset))
    train_data = DataFactory.getData(dataset, train=True)
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    ## get Model
    logger.info('select model: {}, model args: {}'.format(model, model_args))
    model = ModelFactory.getModel(model, **model_args).to(device)
    ## define loss and optimizer
    logger.info('optimizer: {}, learning rate: {}'.format('adam', learn_rate))
    optimizer_function = torch.optim.Adam(model.parameters(), lr=learn_rate)
    ## start train
    logger.info("start train...")
    model.train()
    pbar1 = tqdm(total=epochs)
    loss = 0.0
    for epoch in range(epochs):
        logger.debug("epoch_{} start...".format(epoch))
        pbar1.set_description('epoch_{} start...'.format(epoch))
        pbar2 = tqdm(total=len(train_data_loader))
        for x, y in train_data_loader:
            pbar2.set_description('loss = {:.6f}'.format(loss))
            x = x.to(device)
            optimizer_function.zero_grad()
            output = model(x)
            loss = model.loss_function(output, y)
            logger.info('Epoch: {}, loss = {:.6f}'.format(epoch + 1, loss))
            loss.backward()
            optimizer_function.step()
            pbar2.update()
        pbar2.close()
        pbar1.update()
    pbar1.close()
    ## 展示训练结果
    ## TODO
    ## 测试模型
    logger.info("start test...")
    test_data = DataFactory.getData('mnist', train=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    pbar3 = tqdm(total=len(test_data_loader))
    loss = 0.0
    for x, y in tqdm(test_data_loader, leave=False):
        pbar3.set_description('loss = {:.6f}'.format(loss))
        x = x.to(device)
        output = model(x)
        loss = model.loss_function(output, y)
        logger.info('loss = {:.6f}'.format(loss))
        pbar3.update()
    pbar3.close()
    ## 评估准确率
    x_test, y_test = DataFactory.getValData('mnist')
    y_pred = model(x_test)
    score = model.metrics_function(y_pred, y_test)
    logger.info("acc score is: {}".format(score))
    print("acc score is: {}".format(score))
    ## interact
    y_test = np.argmax(y_test.detach().numpy(), axis=1)
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
    while True:
        print("input idx,then you can view original picture")
        idx = int(input())
        logger.info('id为{}的图片 对应的真实值:{}, 预测值:{}'.format(idx, y_test[idx], y_pred[idx]))
        print('id为{}的图片 对应的真实值:{}, 预测值:{}\n'.format(idx, y_test[idx], y_pred[idx]))
        ImgUtil.img_show(x_test[idx][0] * 256)


