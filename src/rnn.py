# -- Model and Torch Imports -- #
from typing import override
from torch import Tensor, nn as NeuralNetwork, no_grad as NoGrad
from torch import zeros as Zeros

# -- Custom Model Class -- #
from .model import LanguageModel, ActivationMap

class RecurrentNeuralNetwork(LanguageModel):
    def __init__(self, config : dict ):
        super().__init__(config = config)

        try :
            self.LSTM = NeuralNetwork.LSTM(input_size   = self.config['input_size'], 
                                           hidden_size  = self.config['hidden_size'],
                                           num_layers   = self.config['num_layers'],
                                           device       = self.device,
                                           batch_first  = True)
    
            self.hiddenLayers = NeuralNetwork.ModuleList()
            for layer in self.config['linear_layers']:
                self.hiddenLayers.append(NeuralNetwork.Linear(in_features=layer['in'], out_features=layer['out'], device=self.device))
        except KeyError as e:
            print(f"Encounterd KeyError while initializing layers {__file__}/{__name__} : {e}")
            print("Are layers properly defined in config file ?")
        
        try :
            ## Creating the activation functions.
            self.activations = NeuralNetwork.ModuleList([ActivationMap[activation]() for activation in self.config['activations']])
        except KeyError as e:
            print(f"Encounterd KeyError while initializing activation functions {__file__}/{__name__} : {e}")
            print("Are activations properly defined in config file ?")
    
    def initHiddenState(self, batchSize : int) -> Tensor:
        return Zeros(batchSize, self.config['hidden_size'], device=self.device)
    
    def getProbabilityDistribution(self, concatenatedContext : Tensor, dim = 1) -> Tensor:
        with NoGrad():
            output = self.forward(concatenatedContext)
        
        finalToken = output[-1, :]
        probabilityDistribution = NeuralNetwork.Softmax(dim=dim)(finalToken)
        return probabilityDistribution
    
    @override
    def forward(self, x : Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        x, _ = self.LSTM(x)

        for layer, activation in zip(self.hiddenLayers, self.activations):
            x = activation(layer(x))
        
        x = self.hiddenLayers[-1](x)

        return x
