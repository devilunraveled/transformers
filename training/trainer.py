# -- Model and Torch Imports -- #
from math import e
import torch.nn as NeuralNetwork
import torch.optim as Optimizer

from torch import Tensor
from torch.utils.data import DataLoader
from torch import argmax as ArgMax, no_grad as NoGrad

# -- Readability Imports -- #
from typing import Optional, List, Tuple

# -- Custom Classes -- #
from alive_progress import alive_bar

# -- Debugging -- "

class Trainer:
    def __init__(self, 
                 model      : NeuralNetwork.Module,
                 trainData  : DataLoader,
                 validData  : Optional[DataLoader],
                 optimizer  : str,
                 epochs     : int,
                 learning_rate  : float,
                 padIndex       : int = 0
                 ) -> None:
        self.model      = model
        
        if optimizer == 'SGD' :
            self.optimizer  = Optimizer.SGD(model.parameters(), lr=learning_rate)
        elif optimizer == 'Adam' :
            self.optimizer  = Optimizer.Adam(model.parameters(), lr=learning_rate)
        else :
            raise ValueError("Optimizer not supported. Supported optimizers : 'SGD' and 'Adam'.")
        
        self.trainData = trainData
        self.validData = validData
        self.epochs = epochs
        self.padIndex = padIndex
        self.loss = NeuralNetwork.CrossEntropyLoss(ignore_index = padIndex)

    def _trainStep(self, inputs : Tensor, groundTruth : Tensor) -> Tensor:
        output = self.model(inputs)
        # Resize the tensors for Loss calculation.
        output = output.view(-1, output.shape[-1])
        groundTruth = groundTruth.view(-1)
        loss = self.loss(output, groundTruth)
        return loss
    
    def _singleEpochTrain(self, epoch : int, data : DataLoader, propagateLoss : bool = True, progressString : Optional[str] = None) -> Tuple[float, int] :
        overallLoss = 0
        batchSize = 0 
        with alive_bar(len(data), title=f"{progressString} {epoch}", force_tty=True, length=10) as bar:
            currentAvgLoss = 0
            for inputs, groundTruth in data :
                inputs, groundTruth = inputs.to(self.model.device), groundTruth.to(self.model.device)
                
                if propagateLoss :
                    self.optimizer.zero_grad()
                
                loss = self._trainStep(inputs, groundTruth)
                
                if propagateLoss :
                    loss.backward()
                    self.optimizer.step()
                
                overallLoss += loss.item()
                batchSize += 1

                currentAvgLoss = (currentAvgLoss*(batchSize - 1) + loss.item())/(batchSize)
                bar.text = f"\nLoss : {currentAvgLoss:.3f}"

                bar()

        return overallLoss, batchSize

    def train(self, epochs : int, data : Optional[DataLoader] = None , save : bool = False, propagateLoss : bool = True, progressString : Optional[str] = None ) -> List[float]:
        totalLoss = []
        perplexities = []

        if data is None :
            data = self.trainData

        try :
            for i in range(epochs) :
                overallLoss, batchSize = self._singleEpochTrain(i, data, propagateLoss, progressString)
                totalLoss.append(overallLoss/batchSize)
                perplexities.append(pow(e, overallLoss/batchSize))
                print(f"Loss : {overallLoss/batchSize}, Perplexity : {perplexities[-1]}")
            
            return totalLoss
        finally :
            if save :
                self.model.save(f"./ckpts/models/{self.model.__class__.__name__}_{epochs}")
    
    def test(self, data : Optional[DataLoader] = None, title : Optional[str] = None) -> dict:
        totalCorrect = 0
        total = 0
        
        if self.validData is None :
            return {'Accuracy' : 0}
        
        if data is None :
            data = self.validData

        if title is None :
            title = "Validation"

        with NoGrad():
            with alive_bar(len(data), title=title, force_tty=True, length=10) as bar:
                for inputs, groundTruth in data :
                    inputs, groundTruth = inputs.to(self.model.device), groundTruth.to(self.model.device)
                    
                    outputs = self.model(inputs)

                    outputs = outputs.view(-1, outputs.shape[-1])
                    groundTruth = groundTruth.view(-1)
                    predicted = ArgMax(outputs, dim = 1)
                    
                    nonPaddingMask = groundTruth != self.padIndex

                    groundTruth = groundTruth[nonPaddingMask]
                    predicted = predicted[nonPaddingMask]

                    total += groundTruth.size(0)
                    totalCorrect += (predicted == groundTruth).sum().item()

                    bar()
        
        return {'Accuracy' : totalCorrect/total}

    def getPreplexity(self, data : Optional[DataLoader], progressString : Optional[str] = None ) -> float:
        if self.validData is None :
            return 0.0

        if data is None :
            data = self.validData
        
        if progressString is None :
            progressString = "Validation Perplexity"
        
        with NoGrad():
            totalLoss, batchSize = self._singleEpochTrain(0, data, propagateLoss = False, progressString = progressString)
        
        return pow(e, totalLoss/batchSize)
