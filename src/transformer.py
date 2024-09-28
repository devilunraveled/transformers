from typing import override
from torch import nn as NeuralNetwork, cuda as Cuda, Tensor
from torch import triu as UpperTriangle, ones as Ones
from torch import arange as Range, exp as Exponent
from torch import cat as Concatenate, no_grad as NoGrad

from math import log

from .model import LanguageModel


class DecoderBlock(NeuralNetwork.Module):
    def __init__(self, config : dict ):
        super().__init__()
        self.config = config
        self.device = 'cuda' if Cuda.is_available() else 'cpu'
        self.__init_architecture__()

    def __init_architecture__(self):
        # Layer Normalization for the inputs.
        self.norm1 = NeuralNetwork.LayerNorm(normalized_shape=self.config['embedDim'], device = self.device)
        
        self.maskedMultiHeadAttention = NeuralNetwork.MultiheadAttention(embed_dim     = self.config['embedDim'], 
                                                                         num_heads     = self.config['numHeads'], 
                                                                         dropout       = self.config['dropout'], 
                                                                         batch_first   = True,
                                                                         device        = self.device)

        # Layer Normalization for the attention output.
        self.norm2 = NeuralNetwork.LayerNorm(normalized_shape=self.config['embedDim'], device = self.device)
        
        # Apply Linear Layer to the attention output, transforming it to the same dimension as the inputs.
        self.multiLayerPerceptron = NeuralNetwork.Sequential(
            NeuralNetwork.Linear(self.config['embedDim'], self.config['embedDim'] * self.config['mlp_scaler'], device = self.device),
            NeuralNetwork.ELU(),
            NeuralNetwork.Linear(self.config['embedDim'] * self.config['mlp_scaler'], self.config['embedDim'], device = self.device)
        )
    
    @override
    def forward(self, inputs, input_mask = None ):
        _, seqLen, _ = inputs.shape
        
        # Causal Mask
        mask = UpperTriangle( Ones( seqLen, seqLen , device = self.device), 1).bool()

        # Normalize the inputs.
        inputs = self.norm1(input)

        # Self Attention and Skip Connection
        output = self.maskedMultiHeadAttention(inputs, input, input, attn_mask = mask, key_padding_mask = input_mask)[0] + input

        # Normalize the attention outputr.
        output = self.norm2(output)

        # Apply MLP
        output = self.multiLayerPerceptron(output)

        return output

class SinusoidalPositionalEmbedding(NeuralNetwork.Module):
    def __init__(self, dimension : int, device ):
        super().__init__()
        self.dimension = dimension
        self.device = device
    
    @override
    def forward(self, inputs):
        halfDimension = self.dimension // 2
        exponent = log(10000) / (halfDimension - 1)
        embedding = Exponent(Range(halfDimension, device = self.device) * -exponent)
        embedding = inputs[:, None] * embedding[None, :]
        embedding = Concatenate((embedding.sin(), embedding.cos()), dim = -1)
        return embedding

class DecoderOnlyTransformer(LanguageModel):
    def __init__(self, config : dict):
        super().__init__(config)
        self.__init_architecture__()
        self.to(device = self.device)

    def __init_architecture__(self):
        # Positional Embeddings to give postitional information to the model.
        self.positionalEmbeddings = SinusoidalPositionalEmbedding(self.config['embedDim'], self.device)

        # Decoder Stack
        self.decoderStack = NeuralNetwork.ModuleList(
            [DecoderBlock(self.config['blockConfig'][i]) for i in range(self.config['numLayers'])]
        )
        
        # Final MLP for prediction
        self.finalLayer = NeuralNetwork.Linear(self.config['embedDim'], self.config['vocabSize'], device = self.device)
    
    @override
    def forward(self, x ):
        # Masking the pad tokens
        if x.ndim < 3:
            x = x.unsqueeze(0)

        # Get the positions, for this, we need to pass the tensor of indices.
        xIndices = Range(x.shape[-2], device = self.device)
        positionalEmbeddings = self.positionalEmbeddings(xIndices)
        
        positionalEmbeddings = positionalEmbeddings.reshape(1, x.shape[-2], x.shape[-1]).expand(x.shape)
        
        # Add the two to get the final embedding.
        x = x + positionalEmbeddings

        # Decoder Stack
        for decoder in self.decoderStack:
            x = decoder(x)

        # Final Layer
        x = self.finalLayer(x)

        return x

    def getProbabilityDistribution(self, concatenatedContext : Tensor, dim : int = 1) -> Tensor:
        with NoGrad():
            output = self.forward(concatenatedContext)[-1]

        finalTokenDistribution = output[-1, :]
        finalTokenDistribution = NeuralNetwork.Softmax(dim=dim)(finalTokenDistribution)

        return finalTokenDistribution
