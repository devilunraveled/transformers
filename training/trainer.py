# -- Model and Torch Imports -- #
import torch.nn as NeuralNetwork
import torch.optim as Optimizer

from torch import Tensor
from torch.utils.data import DataLoader
from torch import argmax as ArgMax, no_grad as NoGrad

import matplotlib.pyplot as plt
import numpy as np
# -- Readability Imports -- #
from typing import Any, Optional, List, Tuple

# -- Custom Classes -- #
from alive_progress import alive_bar

# -- Debugging -- "

class Trainer:
    def __init__(self, 
                 model      : NeuralNetwork.Module,
                 trainData  : DataLoader,
                 validData  : Optional[DataLoader],
                 optimizer  : str,
                 learning_rate  : float,
                 padIndex       : int = 0,
                 sosIndex       : int = 1,
                 eosIndex       : int = 2,
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
        self.padIndex = padIndex
        self.eosIndex = eosIndex
        self.sosIndex = sosIndex
        self.loss = NeuralNetwork.CrossEntropyLoss(ignore_index = padIndex)

    def _trainStep(self, inputs : Tensor, groundTruth : Tensor) -> Tensor:
        causalInput = groundTruth[:,:-1].contiguous()
        # Replace the EOS tokens with PAD tokens in causalInput.
        causalInput[causalInput == self.eosIndex] = self.padIndex

        causalOutput = groundTruth[:,1:].contiguous()
        output = self.model(inputs, causalInput)
        # Resize the tensors for Loss calculation.
        output = output.view(-1, output.shape[-1])
        causalOutput = causalOutput.view(-1)
        loss = self.loss(output, causalOutput)
        return loss
    

    def _singleEpochTrain(self, epoch: int, data: DataLoader, propagateLoss: bool = True, progressString: Optional[str] = None) -> Tuple[float, int]:
        iterationWiseLoss = []
        smootherLoss = []
        overallLoss = 0
        batchSize = 0
        
        # Initialize the plot
        plotDuration = 50
        plt.ion()  # Turn on interactive mode
        plt.style.use("dark_background") # Darkk Mode.
        fig, ax = plt.subplots()
        line, = ax.plot([], marker='o', linestyle='-', color='tan', label='Loss')
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.set_title(f"{progressString} {epoch + 1}")
        ax.legend()

        # Set up initial limits
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)  # Assumed loss values; will auto-adjust

        with alive_bar(len(data), title=f"{progressString} {epoch + 1}", force_tty=True, length=10) as bar:
            currentAvgLoss = 0
            for batch_idx, (inputs, groundTruth) in enumerate(data):
                inputs, groundTruth = inputs.to(self.model.device), groundTruth.to(self.model.device)
                
                if propagateLoss:
                    self.optimizer.zero_grad()

                loss = self._trainStep(inputs, groundTruth)

                if propagateLoss:
                    loss.backward()
                    self.optimizer.step()

                currLoss = loss.item()
                overallLoss += currLoss
                batchSize += 1

                currentAvgLoss = overallLoss / batchSize

                iterationWiseLoss.append(currentAvgLoss)
                # Update the bar text with the current average loss
                bar.text = f"\nLoss : {currentAvgLoss:.3f}"
                bar()

                # Update the plot every few iterations to reduce overhead
                if batch_idx % plotDuration == 0 or batch_idx == len(data) - 1:
                    smootherLoss.append(np.mean(iterationWiseLoss[-plotDuration:]))
                    
                    line.set_ydata(smootherLoss)
                    xValues = [plotDuration * i for i in range(len(smootherLoss))]
                    line.set_xdata(xValues)
                    
                    # Dynamically adjust x-axis limits based on iteration count
                    maxXVal = max(10, max(xValues))
                    ax.set_xlim(0, maxXVal)
                    maxYVal = min(11, max(smootherLoss))
                    ax.set_ylim(max(0,min(5, min(smootherLoss) - 1)), maxYVal)  # Assumed loss values; will auto-adjust

                    # Adjust axis dynamically for y-data as needed
                    ax.relim()
                    ax.autoscale_view()

                    plt.draw()
                    plt.pause(0.001)

        plt.ioff()  # Turn off interactive mode
        plt.close(fig)

        return overallLoss, batchSize

    def train(self, epochs : int, data : Optional[DataLoader] = None , save : bool = False, propagateLoss : bool = True, progressString : Optional[str] = None ) -> List[float]:
        totalLoss = []

        if data is None :
            data = self.trainData

        try :
            for i in range(epochs) :
                overallLoss, batchSize = self._singleEpochTrain(i, data, propagateLoss, progressString)
                totalLoss.append(overallLoss/batchSize)
                print(f"Loss : {overallLoss/batchSize}")
            
            return totalLoss
        finally :
            if save :
                self.model.save(f"./ckpts/models/{self.model.__class__.__name__}_{epochs}.pt")

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
                    
                    causalInput = groundTruth[:,:-1]
                    causalInput[causalInput == self.eosIndex] = self.padIndex
                    causalOutput = groundTruth[:,1:]

                    outputs = self.model(inputs, causalInput)
                    outputs = outputs.view(-1, outputs.shape[-1])

                    groundTruth = causalOutput.reshape(-1)
                    predicted = ArgMax(outputs, dim = 1)
                    nonPaddingMask = groundTruth != self.padIndex

                    groundTruth = groundTruth[nonPaddingMask]
                    predicted = predicted[nonPaddingMask]

                    total += groundTruth.size(0)
                    totalCorrect += (predicted == groundTruth).sum().item()

                    bar()
        
        return {'Accuracy' : totalCorrect/total}

    def getModelResponse(self, vocabBidict : Any, data : Optional[DataLoader] = None, title : Optional[str] = None):
        assert self.validData is not None

        if data is None :
            data = self.validData

        if title is None :
            title = "Validation"
        
        outputSentences = []
        goldenSentences = []

        with NoGrad():
            with alive_bar(len(data), title=title, force_tty=True, length=10) as bar:
                for inputs, groundTruth in data :
                    inputs, groundTruth = inputs.to(self.model.device), groundTruth.to(self.model.device)
                    
                    causalInput = groundTruth[:,:-1]
                    causalInput[causalInput == self.eosIndex] = self.padIndex
                    causalOutput = groundTruth[:,1:].cpu().detach().numpy()

                    outputs = self.model(inputs, causalInput)
                    predicted = ArgMax(outputs, dim = 2).cpu().detach().numpy()

                    for i in range(predicted.shape[0]) :
                        outputSentence = [vocabBidict.inv[x] for x in predicted[i] if x != self.padIndex and x != self.eosIndex and x != self.sosIndex]
                        causalSentence = [vocabBidict.inv[x] for x in causalOutput[i] if x != self.padIndex and x != self.eosIndex and x != self.sosIndex]
                        outputSentences.append(outputSentence)
                        goldenSentences.append(causalSentence)
                    bar()

        return outputSentences, goldenSentences 
