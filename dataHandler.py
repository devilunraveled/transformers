from numpy.typing import NDArray
from torch.utils.data import Dataset
from torch import from_numpy as FromNumpy

from typing import List, Callable
from utils import createDataPoints
from numpy import concatenate as Concat, array as Array

class ArtificialNeuralNetworkDataset(Dataset):
    def __init__(self, data : List[List[NDArray]], tokens : List[List[str]], getIndex : Callable[[str], int],  window : int = 5 ):
        self.datapoints = createDataPoints(data, tokens, getIndex, window, dataPointType='ANN')

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        context = Concat(self.datapoints[idx].context, dtype='float32')
        nextToken = Array(self.datapoints[idx].nextToken, dtype='int64')
        return context, nextToken

class RecurrentNeuralNetworkDataset(Dataset):
    def __init__(self, data : List[List[NDArray]], tokens : List[List[str]], getIndex : Callable[[str], int],  window : int = 5 ):
        self.datapoints = createDataPoints(data, tokens, getIndex, window, dataPointType='RNN')

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        context = FromNumpy(Array(self.datapoints[idx].sentence, dtype='float32'))
        nextToken = FromNumpy(Array(self.datapoints[idx].nextTokens, dtype='int64'))
        return context, nextToken
