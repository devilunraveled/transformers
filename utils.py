from torch.nn.utils.rnn import pad_sequence as PadSequence

from collections import namedtuple as NamedTuple
from typing import Callable, List, Optional

from numpy.typing import NDArray

ArtificialNeuralNetworkDataPoint = NamedTuple('ANNDataPoint', [ 'context', 'nextToken' ])
RecurrentNeuralNetworkDataPoint = NamedTuple('RNNDataPoint', [ 'sentence', 'nextTokens' ])

def createDatapointsForAnnFromSentence(sentence : List[NDArray], tokens : List[str], getIndex : Callable[[str], int], window : int = 5 ):
    datapoints = []
    for i in range(window, len(sentence)-window):
        datapoints.append(ArtificialNeuralNetworkDataPoint(sentence[i-window:i], getIndex(tokens[i])))
    return datapoints

def createDatapointsForRnnFromSentence(sentence : List[NDArray], tokens : List[str], getIndex : Callable[[str], int], window : int = 5 ):
    return RecurrentNeuralNetworkDataPoint(sentence[window:-window-1], [getIndex(tokens[i]) for i in range(window + 1, len(sentence) - window)])

def createDataPoints(data : List[List[NDArray]], allTokens : List[List[str]] , getIndex : Callable[[str], int], window : int = 5, dataPointType : Optional[str] = 'ANN' ):
    datapoints = []
    for sentence, tokens in zip(data, allTokens):
        if dataPointType == 'ANN':
            datapoints.extend(createDatapointsForAnnFromSentence(sentence, tokens, getIndex, window))
        elif dataPointType == 'RNN':
            datapoints.append(createDatapointsForRnnFromSentence(sentence, tokens, getIndex, window))
    return datapoints

def dataCollater(batch, padToken):
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=False)
    sentences, nextTokens = zip(*batch)
    paddedSentences = PadSequence(sentences, padding_value = padToken, batch_first=True)
    paddedNextTokens = PadSequence(nextTokens, padding_value = padToken, batch_first=True)

    return paddedSentences, paddedNextTokens
