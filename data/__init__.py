"""
   Author: hxt
   Created: 2024/3/13 
"""
import logging
from data.MNIST import MNIST

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DataFactory:

    def __init__(self):
        pass

    @staticmethod
    def getData(dataset: str, train: bool = True) -> Dataset:
        if dataset == 'mnist':
            return MNIST(train=train)
        else:
            logger.error("DatasetFactory received an unknown dataset %s", dataset)
            raise Exception(f"Unrecognized Dataset: {dataset}")

    @staticmethod
    def getValData(dataset: str):
        if dataset == 'mnist':
            return MNIST(train=False).getValData()
        else:
            logger.error("DatasetFactory received an unknown dataset %s", dataset)
            raise Exception(f"Unrecognized Dataset: {dataset}")