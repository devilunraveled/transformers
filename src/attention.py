"""
This module implements the attention mechanisms for the Transformer Architecture.
"""

from typing import Optional, override
from torch import nn as NeuralNetwork
from torch.functional import Tensor
from torch import exp as Exponent, ones as Ones

def getAttentionScores(query, key):
    dim = query.shape[-1]
    return query @ key.transpose(-2, -1) / (dim ** 0.5)

class MultiHeadAttention(NeuralNetwork.Module):
    def __init__(self, inputDim : int, numHeads : int, dropout : float, device : Optional[str] = None):
        super().__init__()
        assert ( inputDim % numHeads ) == 0
        self.inputDim = inputDim
        self.numHeads = numHeads
        self.embedDim = inputDim // numHeads
        self.dropout = dropout
        self.device = device
        self.__init_architecture__()

    def __init_architecture__(self):
        self.QueryW = NeuralNetwork.Linear(self.embedDim, self.embedDim, device = self.device)
        self.KeyW   = NeuralNetwork.Linear(self.embedDim, self.embedDim, device = self.device)
        self.ValueW = NeuralNetwork.Linear(self.embedDim, self.embedDim, device = self.device)

    @override
    def forward(self, 
                keyRepresentation   : Tensor, 
                queryRepresentation : Tensor, 
                valueRepresentation : Tensor, 
                attn_mask           : Optional[Tensor] = None):
        try :
            assert queryRepresentation.shape[-1] == keyRepresentation.shape[-1] == valueRepresentation.shape[-1]
        except AssertionError :
            print("Recieved embeddingSize does not match the from the passed configuration.")
            print(queryRepresentation.shape, keyRepresentation.shape, valueRepresentation.shape)
        
        # Reshape the inputs, keys and values to fill 
        # from (BatchSize, SeqLen, InputDim) to (BatchSize, SeqLen, NumHeads, EmbedDim)
        batchSize, seqLen, _ = keyRepresentation.shape
        
        numHeads = self.numHeads
        embedDim = self.embedDim
        key     = keyRepresentation.reshape(batchSize, seqLen, numHeads, embedDim)
        query   = queryRepresentation.reshape(batchSize, seqLen, numHeads, embedDim)
        value   = valueRepresentation.reshape(batchSize, seqLen, numHeads, embedDim)

        key = key.transpose(-2, -3)
        query = query.transpose(-2, -3)
        value = value.transpose(-2, -3)

        key = self.KeyW(key)
        query = self.QueryW(query)
        value = self.ValueW(value)

        scores = getAttentionScores(query, key)
        
        if attn_mask is None:
            attn_mask = Ones(scores.shape[-2:], device = self.device)

        # Applying Softmax to the scores
        # Step 1: Subtract the maximum value from the scores for numerical stability
        max_scores = scores.max(dim=-1, keepdim=True)[0]
        stable_scores = scores - max_scores

        # Step 2: Compute softmax in a numerically stable way
        scores = Exponent(stable_scores)

        # scores = scores.masked_fill(attn_mask == 0, 0)
        scores = scores / scores.sum(dim=-1, keepdim=True)
        
        output = scores @ value

        output = output.transpose(-2, -3)
        # Reconstruct the original shape from (BatchSize, SeqLen, NumHeads, EmbedDim) to (BatchSize, SeqLen, InputDim)
        output = output.reshape(output.shape[0], output.shape[1], self.inputDim)
        return output
