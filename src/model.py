from torch import nn as NeuralNetwork
from torch import save as Save, load as Load
from torch import device as Device, cuda as Cuda

from os import path as Path
from json import load as LoadJson, dump as DumpJson

from typing import Optional, override

from .model_config import CHECKPOINTS

ActivationMap : dict = {
    'sigmoid' : NeuralNetwork.Sigmoid,
    'tanh' : NeuralNetwork.Tanh,
    'relu' : NeuralNetwork.ReLU,
    'softmax' : NeuralNetwork.Softmax,
    'leaky_relu' : NeuralNetwork.LeakyReLU
}

class LanguageModel(NeuralNetwork.Module):
    @classmethod
    def load(cls, path : str):
        if Path.exists(f"{path}{CHECKPOINTS['model_extenstion']}") :
            if Path.exists(f"{path}{CHECKPOINTS['config_extenstion']}") :
                with open(f'{path}{CHECKPOINTS["config_extenstion"]}', 'r', encoding='utf-8') as f:
                    config = LoadJson(f)
                
                model = cls(config)
                model.load_state_dict(Load(f"{path}{CHECKPOINTS['model_extenstion']}", weights_only=True))
                return model
            else :
                raise FileNotFoundError(f"Config File not found at path: `{path}{CHECKPOINTS['config_extenstion']}`")
        else:
            raise FileNotFoundError(f"File not found at path: `{path}`")

    def save(self, path : str):
        Save(self.state_dict(), f"{path}{CHECKPOINTS['model_extenstion']}")
        with open(f'{path}{CHECKPOINTS["config_extenstion"]}', 'w', encoding='utf-8') as f:
            DumpJson(self.config, f)
        print(f"Model saved at : {path}")
    
    
    def __init__(self, config : dict, name : Optional[str] = None):
        super().__init__()
        self.config = config
        if name is None :
            self.name = 'LanguageModel'
        else :
            self.name = name

        self.device = Device('cuda' if Cuda.is_available() else 'cpu')
        self.to(self.device)
    
    @override 
    def forward(self, x) :
        print(x)
        raise NotImplementedError
