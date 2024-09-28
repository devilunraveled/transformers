from dataset.data_config import DataConfig
from .data_utils import loadFile, removeSpecialCharacters, saveFile

from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split as ParitionSplit
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

from bidict import bidict

from .data_config import DataConfig

class Partition:
    def __init__(self, name : str , path : str, vocab : bidict, data : List[List[str]] ):
        """
        Base Class for Partition, which can be train, 
        test or valid. The vocab is shared for all 
        paritions of the same dataset.
        """
        self.name   : str = name
        self.path   : str = path

        self.data   : List[List[str]] = data
        self.vocab  : bidict = vocab 
    
    def getData(self) -> List[List[str]]:
        return self.data

    def savePartition(self) -> None:
        saveFile(self.path, '\n'.join( [ ' '.join(sent) for sent in self.data] ))

class Data:
    def __init__(self, name : str , path : str ):
        """
        Data Class for preprocessing pipeline.
        """
        self.name   : str = name
        self.path   : str = path 
        self.tokenizer : Tokenizer = Tokenizer()

        self.vocab  : bidict[str, int] = bidict()
        self._raw_data : str = ""
        self.data      : str = ""
        self.tokenizedData : List[List[str]] = [['']]
        self.partitions : Dict[str, Partition] = {}

    def loadData(self) -> str:
        self._raw_data = loadFile(self.path)
        return self._raw_data

    def sanitizeText(self) -> None :
        self.data = self._raw_data.replace('\n', ' ').strip()
        self.tokenizedData=[ self.tokenizer.tokenizeWords(removeSpecialCharacters(text)) for text in self.tokenizer.tokenizeSentences(self.data)]

    def getVocab(self) -> bidict[str, int] :
        vocab = Counter([DataConfig['customTokens']['unknown']] + [word for sentence in self.tokenizedData for word in sentence])
        self.vocab = bidict((word, i) for i, word in zip(range(len(vocab)), vocab.keys()))
        return self.vocab
    
    def getIndex(self, token : str) -> int:
        return self.vocab.get(token, self.vocab[DataConfig['customTokens']['unknown']])

    def createPartitions(self, partitionSize : Tuple[float, float, float]) -> Dict[str, Partition]:
        """
        Create partitions with ratio => train : test : valid
        as present in partitionSize[0], partitionSize[1], partitionSize[2].
        """
        trainSize, testSize, validSize = partitionSize

        ## Split data into partitions
        trainSplit, rest = ParitionSplit(self.tokenizedData, test_size = (testSize + validSize)/(trainSize + testSize + validSize))
        testSplit, validSplit = ParitionSplit(rest, test_size = validSize/(testSize + validSize))

        self.partitions = {
            'train' : Partition(DataConfig['train']['name'], DataConfig['train']['path'], self.vocab, trainSplit),
            'test'  : Partition(DataConfig['test']['name'], DataConfig['test']['path'], self.vocab, testSplit),
            'valid' : Partition(DataConfig['valid']['name'], DataConfig['valid']['path'], self.vocab, validSplit),
        }

        return self.partitions

class Tokenizer :
    def __init__(self) -> None:
        self.wordTokenizer = word_tokenize
        self.sentTokenizer = sent_tokenize

    def tokenizeSentences(self, text : str) -> List[str]:
        return self.sentTokenizer(text)

    def tokenizeWords(self, text : str) -> List[str]:
        padding = [ DataConfig['customTokens']['pad'] for _ in range(DataConfig['padding']) ]
        return padding + [DataConfig['customTokens']['start']] + self.wordTokenizer(text) + [DataConfig['customTokens']['end']] + padding
