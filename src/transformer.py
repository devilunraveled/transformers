from typing import override
from torch import nn as NeuralNetwork, cuda as Cuda, Tensor
from torch import triu as UpperTriangle, ones as Ones
from torch import arange as Range, exp as Exponent
from torch import cat as Concatenate, no_grad as NoGrad

from math import log

from .model import LanguageModel
from .attention import MultiHeadAttention

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

class DecoderBlock(NeuralNetwork.Module):
    def __init__(self, config : dict ):
        super().__init__()
        self.config = config
        self.device = 'cuda' if Cuda.is_available() else 'cpu'
        self.__init_architecture__()

    def __init_architecture__(self):
        # Masked Attention to Employ Causality during Next token Prediction
        self.maskedMultiHeadAttention = MultiHeadAttention( inputDim    = self.config['inputDim'], 
                                                            numHeads    = self.config['numHeads'],
                                                            dropout     = self.config['dropout'],
                                                            device      = self.device)
        
        # Layer Normalization after the application of Masked Multi-Head Attention
        self.norm1 = NeuralNetwork.LayerNorm(normalized_shape=self.config['inputDim'], device = self.device)
        
        # Cross Attention, with Key, Value from Encoder's last layer.
        self.crossMultiHeadAttention  = MultiHeadAttention( inputDim    = self.config['inputDim'], 
                                                            numHeads    = self.config['numHeads'],
                                                            dropout     = self.config['dropout'],
                                                            device      = self.device)
        
        # Layer Normalization for the attention output.
        self.norm2 = NeuralNetwork.LayerNorm(normalized_shape=self.config['inputDim'], device = self.device)

        # Apply Linear Layer to the attention output, transforming it to the same dimension as the inputs.
        self.multiLayerPerceptron = NeuralNetwork.Sequential(
            NeuralNetwork.Linear(self.config['inputDim'], self.config['inputDim'] * self.config['mlp_scaler'], device = self.device),
            NeuralNetwork.ELU(),
            NeuralNetwork.Linear(self.config['inputDim'] * self.config['mlp_scaler'], self.config['outputDim'], device = self.device)
        )

        # Layer Normalization for the output of the layer.
        self.norm3 = NeuralNetwork.LayerNorm(normalized_shape = self.config['inputDim'], device = self.device)
    
    @override
    def forward(self, inputs, encoderOutput):
        _, seqLen, _ = inputs.shape
        
        # Causal Mask
        mask = UpperTriangle( Ones( seqLen, seqLen , device = self.device), 1).bool()

        # Self Attention and Skip Connection
        outputs = self.maskedMultiHeadAttention(keyRepresentation   = inputs, 
                                                queryRepresentation = inputs, 
                                                valueRepresentation = inputs, 
                                                attn_mask           = mask ) + inputs

        # Normalize the attentions
        outputs = self.norm1(outputs)
        
        # Employ Cross Attention
        outputs = self.crossMultiHeadAttention(encoderOutput, outputs, encoderOutput) + outputs # Key, Query, Value

        # Normalize the attention output.
        outputs = self.norm2(outputs)

        # Apply MLP
        outputs = self.multiLayerPerceptron(outputs) + outputs
        
        # Final Layer Normalization
        outputs = self.norm3(outputs)

        return outputs

class EncoderBlock(NeuralNetwork.Module):
    def __init__(self, config : dict ):
        super().__init__()
        self.config = config
        self.device = 'cuda' if Cuda.is_available() else 'cpu'
        self.__init_architecture__()

    def __init_architecture__(self):
        # No masking is done since the input is visible
        self.multiHeadAttention = MultiHeadAttention(inputDim    = self.config['inputDim'], 
                                                     numHeads    = self.config['numHeads'],
                                                     dropout     = self.config['dropout'],
                                                     device      = self.device)
        
        # Layer Normalizations
        self.norm1 = NeuralNetwork.LayerNorm(normalized_shape = self.config['inputDim'], device = self.device)

        # Applying Linear Layer to the outputs from the layer normalization of the attention outputs.
        self.multiLayerPerceptron = NeuralNetwork.Sequential(
            NeuralNetwork.Linear(self.config['inputDim'], self.config['inputDim'] * self.config['mlp_scaler'], device = self.device),
            NeuralNetwork.ELU(),
            NeuralNetwork.Linear(self.config['inputDim'] * self.config['mlp_scaler'], self.config['outputDim'], device = self.device)
        )
        
        # Final Layer Normalization
        self.norm2 = NeuralNetwork.LayerNorm(normalized_shape = self.config['inputDim'], device = self.device)
    
    @override
    def forward(self, inputs):
        outputs = self.multiHeadAttention(keyRepresentation = inputs, 
                                          queryRepresentation = inputs, 
                                          valueRepresentation = inputs)
        
        # Layer Normalization after computing Attention Output
        outputs = self.norm1(outputs)

        # Pass the outputs through the MLP
        outputs = self.multiLayerPerceptron(outputs)

        # Final Layer Normalization
        outputs = self.norm2(outputs)

        return outputs

class EncoderDecoderTransformer(LanguageModel):
    def __init__(self, modelConfig ):
        super().__init__(modelConfig)
        self.__init_architecture__()
        self.to(self.device)

    def __init_architecture__(self):
        # Embedding Layers
        self.EncoderEmbeddings = NeuralNetwork.Embedding(num_embeddings =self.config['num_enc_embeddings'],
                                                         embedding_dim  =self.config['inputDim'],
                                                         padding_idx = self.config['padding_idx'] )
        
        self.DecoderEmbeddings = NeuralNetwork.Embedding(num_embeddings =self.config['num_dec_embeddings'],
                                                         embedding_dim  =self.config['outputDim'],
                                                         padding_idx = self.config['padding_idx'] )
        
        # Positional Embeddings
        self.EncoderPositionalEmbeddings = SinusoidalPositionalEmbedding(self.config['inputDim'], self.device)
        self.DecoderPositionalEmbeddings = SinusoidalPositionalEmbedding(self.config['outputDim'], self.device)

        # Encoder Stack
        self.encoderStack = NeuralNetwork.Sequential(
            *[EncoderBlock(self.config['encoderBlockConfig'][i]) for i in range (self.config['numEncoderLayers']) ]
        )

        # Decoder Stack
        self.decoderStack = NeuralNetwork.ModuleList(
            [DecoderBlock(self.config['decoderBlockConfig'][i]) for i in range (self.config['numDecoderLayers']) ]
        )

        self.classificationLayer = NeuralNetwork.Linear(self.config['modelOutputDimension'], self.config['outputVocabSize'] )

    @override
    def forward(self, inputs, outputs ):
        if inputs.dim() < 2 : # No Batching.
            inputs = inputs.unsqueeze(0)
        
        # Prepare the positional embeddings.
        inputIndices = Range(inputs.shape[-2], device = self.device)
        positionalInformation = self.EncoderPositionalEmbeddings(inputIndices)
        
        # Expand the positional embeddings for the entire batch.
        positionalInformation = positionalInformation.reshape(positionalInformation.shape[0], 1, positionalInformation.shape[1]).expand(-1, inputs.shape[1], -1)
        # Get embeddings from Embedding Lookup for Encoder.
        inputs = self.EncoderEmbeddings(inputs)
        
        # Add the positional encodings to the embeddings.
        inputs = inputs + positionalInformation

        # Passing all the values to the decoder stack.
        encoderOutput = self.encoderStack(inputs)

        # Preparing Decoder Input.
        if outputs.dim() < 2 : # No Batching
            outputs = outputs.unsqueeze(0)
        
        # Prepare the positional embeddings for the outputs.
        outputIndices = Range(outputs.shape[-2], device = self.device)
        positionalInformation = self.DecoderPositionalEmbeddings(outputIndices)
        positionalInformation = positionalInformation.reshape(positionalInformation.shape[0], 1, positionalInformation.shape[1]).expand(-1, outputs.shape[1], -1)

        outputs = self.DecoderEmbeddings(outputs)

        # Add the positinal information to the embeddings.
        outputs = outputs + positionalInformation

        # Passing the outputs through the decoders.
        for decoder in self.decoderStack :
            outputs = decoder(outputs, encoderOutput)
        
        # Apply final linear layer to get vocab distribution.
        output = self.classificationLayer(outputs)

        return output
