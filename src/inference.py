from typing import List, Any
from alive_progress import alive_it
from torch import Tensor, from_numpy as FromNumpy, argmax as ArgMax
from numpy import stack as Stack, concatenate as Concat
from torch.types import Number

from .model import LanguageModel
from bidict import bidict

class Inference:
    @classmethod
    def from_model(cls, model : LanguageModel, embeddingMap : Any, vocab : bidict, window : int = 5) -> object:
        return cls(model=model, embeddingMap=embeddingMap, vocab=vocab, window=window)
    
    def __init__(self, model : LanguageModel, embeddingMap : Any, vocab : bidict, window : int = 5 ) -> None:
        self.model = model
        self.embeddingMap = embeddingMap
        self.vocab = vocab
        self.window = window
        self.Func = Concat if self.model.__class__.__name__ == 'ArtificialNeuralNetwork' else Stack
        self.factor = 1 if self.model.__class__.__name__ == 'ArtificialNeuralNetwork' else 0

    def nextWord(self, concatenatedContext : Tensor ) -> tuple[Number, str]:
        concatenatedContext = concatenatedContext.to(self.model.device)
        probabilityDistribution = self.model.getProbabilityDistribution(concatenatedContext=concatenatedContext)
        nextToken = ArgMax(probabilityDistribution).item()
        return nextToken, self.vocab.inv[nextToken]

    def predict(self, context : List[str] ) -> str:
        try :
            context = context[-self.window:]
            concatenatedContext = FromNumpy(self.Func([self.embeddingMap[token] for token in context], dtype='float32')).to(self.model.device)
            return self.nextWord(concatenatedContext=concatenatedContext)[1]
        except KeyError as e:
            print(f"Encounterd KeyError while predicting next word {__file__}/{__name__} : {e}")
            return ""

    def calculatePerplexity(self, model : LanguageModel, sentence : List[str], padded : bool = True) -> float:
        sentenceProbability = 1e0
        padding = self.window
        if not padded :
            sentence = ['<UNK>']*padding + ['<SOS>'] + sentence + ['<EOS>'] + ['<PAD>']*padding

        for i in range(padding, len(sentence) - padding) :
            context = sentence[i-self.window:i]
            inputs = FromNumpy(self.Func([self.embeddingMap[token] for token in context], dtype='float32')).to(model.device)
            outputDistribution = model.getProbabilityDistribution(concatenatedContext=inputs, dim=self.factor).squeeze().detach().cpu().numpy()
            probability = outputDistribution[self.vocab[sentence[i]]]
            
            sentenceProbability *= probability
        
        perplexity = pow(1/(sentenceProbability + 1e-60), 1/ ( len(sentence) - ( 1 + self.factor)*padding ))
        return perplexity

    def getPerplexityScores(self, data : List[List[str]]) :
        scores = [self.calculatePerplexity(model=self.model, sentence=sentence) for sentence in alive_it(data, force_tty=True, length=10)]
        return list(scores)
