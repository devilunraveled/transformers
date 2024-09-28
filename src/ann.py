# -- Model and Torch Imports -- #
from typing import override
from torch import Tensor, nn as NeuralNetwork, no_grad as NoGrad

# -- Custom Model Class -- #
from .model import LanguageModel, ActivationMap

class ArtificialNeuralNetwork(LanguageModel):
    def __init__(self, config : dict ):
        super().__init__(config)
        
        try :
            ## Creating the weight matrices.
            self.layers = NeuralNetwork.ModuleList()
            for layer in self.config['layers']:
                self.layers.append(NeuralNetwork.Linear(in_features=layer['in'], out_features=layer['out'], device=self.device))
                print(f"Initialized layer {layer['in']} -> {layer['out']}")
        except KeyError as e:
            print(f"Encounterd KeyError while initializing layers {__file__}/{__name__} : {e}")
            print("Are layers properly defined in config file ?")

        try :
            ## Creating the activation functions.
            self.activations = NeuralNetwork.ModuleList()
            for activation in self.config['activations']:
                if activation in ActivationMap :
                    self.activations.append(ActivationMap[activation]())
                else:
                    raise KeyError(f"Activation {activation} not supported")
        except KeyError as e:
            print(f"Encounterd KeyError while initializing activation functions {__file__}/{__name__} : {e}")
            print("Are activations properly defined in config file ?")
    
    @override
    def forward(self, x : Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        
        x = self.layers[-1](x)

        return x

    def getProbabilityDistribution(self, concatenatedContext : Tensor, dim : int = 1) -> Tensor:
        with NoGrad():
            return NeuralNetwork.Softmax(dim=dim)(self.forward(concatenatedContext))
