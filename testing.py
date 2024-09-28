# %%
# %load_ext autoreload
# %autoreload 2

# %% [md]
"""
# Feature Testing
"""


# %% [md]
"""
### Data Preparation and Preprocessing.
"""

# %%
from config import EMBEDDING_SIZE, ANN_MODEL_SAMPLE_CONFIG, TRAINER_SAMPLE_CONFIG
from dataset.data import Data
from dataset.data_config import DataConfig

# %%
padToken = DataConfig['customTokens']['pad']

# %%
data = Data('Auguste_Maquest', './corpus/Auguste_Maquet.txt')

# %%
data.loadData()
data.sanitizeText()
vocabSize = len(data.getVocab())

# %%
len(data.tokenizedData)

# %%
partitions = data.createPartitions((0.7, 0.2, 0.1))

# %%
partitions['train'].savePartition()
partitions['test'].savePartition()
partitions['valid'].savePartition()

# %%
print(f"Train Size : {len(partitions['train'].data)}\nTest Size : {len(partitions['test'].data)}\nValid Size : {len(partitions['valid'].data)}")

# %% [md]
"""
### Pre Trained Embeddings
"""

# %%
from embeddings.pretrained import PreTrainedEmbeddings

# %%
embeddings = PreTrainedEmbeddings.from_pretrained('./ckpts/w2v/w2v.model')

# %%
trainEmbeddings = embeddings(partitions['train'].getData())
testEmbeddings = embeddings(partitions['test'].getData())
validEmbeddings = embeddings(partitions['valid'].getData())

# %% [md]
"""
### Data Loader
"""

# %%
from torch.utils.data import DataLoader
from dataHandler import ArtificialNeuralNetworkDataset

# %%
from training.trainer import Trainer
from src.ann import ArtificialNeuralNetwork
from src.inference import Inference
# %%
# config = {
#     'layers' : [
#         {
#             'in' : EMBEDDING_SIZE*5,
#             'out' : EMBEDDING_SIZE
#         },
#         {
#             'in' : EMBEDDING_SIZE,
#             'out' : vocabSize
#         }
#     ],
#     'activations' : [
#         'leaky_relu',
#     ]
# }
# model = ArtificialNeuralNetwork(config)
model = ArtificialNeuralNetwork.load('./ckpts/models/ArtificialNeuralNetwork_15_0.1805')

# %%
print(f"Model on {model.device}")

# %%
trainDataloader = DataLoader(ArtificialNeuralNetworkDataset(trainEmbeddings, tokens = partitions['train'].getData(), getIndex = data.getIndex), batch_size=64, shuffle=False, num_workers = 16, pin_memory = True)
testDataloader = DataLoader(ArtificialNeuralNetworkDataset(testEmbeddings, tokens = partitions['test'].getData(), getIndex = data.getIndex), batch_size=10, shuffle=False, num_workers = 4, pin_memory = True)
validDataloader = DataLoader(ArtificialNeuralNetworkDataset(validEmbeddings, tokens = partitions['valid'].getData(), getIndex = data.getIndex), batch_size=10, shuffle=False, num_workers = 4, pin_memory = True)

# %% [md]
"""
### Training Models
"""

# %%
trainer = Trainer(model,trainData = trainDataloader, validData = validDataloader, **TRAINER_SAMPLE_CONFIG)

# %%
sampleSentence = partitions['test'].getData()[1]
print(sampleSentence)

# %%
for epoch in range(10):
    trainer.train(epochs = 1, data = trainDataloader, save = False, propagateLoss = True, progressString = f"Training")
    trainer.train(epochs = 1, data = testDataloader, save = False, propagateLoss = False, progressString = f"Testing")
    trainer.train(epochs = 1, data = validDataloader, save = False, propagateLoss = False, progressString = f"Validating")
    print("-------"*10)
# %%
trainer.train(epochs = 0, data = trainDataloader, save = True) 

trainer.test(data = testDataloader)

# %% [md]
"""
## Using a pre-trained model.
"""

# %%
inference = Inference(model, embeddings, data.getVocab())
inference.predict(['this', 'is', 'the', 'first', 'time'])

# %%
inference.calculatePerplexity(model, ['this', 'is', 'the', 'first', 'time', 'of', 'the', 'day'], padded = False)

# %%
perplexityScores = inference.getPerplexityScores( data = partitions['test'].getData() )

# %%
print(sum(perplexityScores)/len(perplexityScores))

# %% [md]
"""
# LSTM Testing
"""

# %%
from src.rnn import RecurrentNeuralNetwork
from dataHandler import RecurrentNeuralNetworkDataset
from config import LSTM_MODEL_SAMPLE_CONFIG
from utils import dataCollater

# %%
LSTM_MODEL_SAMPLE_CONFIG['linear_layers'][-1]['out'] = vocabSize

# %%
padValue = data.getIndex(padToken)

# %%
trainData = DataLoader(RecurrentNeuralNetworkDataset(trainEmbeddings, tokens = partitions['train'].getData(), getIndex = data.getIndex), batch_size=8 , shuffle=False, num_workers = 4, pin_memory = True, collate_fn = lambda x : dataCollater(x, padToken=padValue))
validData = DataLoader(RecurrentNeuralNetworkDataset(validEmbeddings, tokens = partitions['valid'].getData(), getIndex = data.getIndex), batch_size=4, shuffle=False, num_workers = 4, pin_memory = True, collate_fn = lambda x : dataCollater(x, padToken=padValue))
testData = DataLoader(RecurrentNeuralNetworkDataset(testEmbeddings, tokens = partitions['test'].getData(), getIndex = data.getIndex), batch_size=4, shuffle=False, num_workers = 4, pin_memory = True, collate_fn = lambda x : dataCollater(x, padToken=padValue))
# %%
# lstmModel = RecurrentNeuralNetwork(LSTM_MODEL_SAMPLE_CONFIG)
lstmModel = RecurrentNeuralNetwork.load('./ckpts/models/RecurrentNeuralNetwork_10_0.8021')

# %%
print(f"Model on {lstmModel.device}")

# %%
trainer = Trainer(lstmModel, trainData, validData, padIndex = padValue, **TRAINER_SAMPLE_CONFIG)

# %%
# trainer.train(epochs=10)

# %%
trainer.test()

# %% [md]
"""
## Transformer Testing
"""

# %%
from src.transformer import DecoderOnlyTransformer
from config import TRANSFORMER_CONFIG
from torch import from_numpy as FromNumpy

TRANSFORMER_CONFIG['padToken'] = FromNumpy(embeddings['padToken']).unsqueeze(0).unsqueeze(0)

# %%
transformer = DecoderOnlyTransformer(TRANSFORMER_CONFIG)
print(f"Transformer on {transformer.device}")

# %%
trainer = Trainer(transformer, trainData, validData, **TRAINER_SAMPLE_CONFIG)

# %%
trainer.train(epochs=10, progressString = "Training")

# %%
trainer.getPreplexity(data = testData, progressString = "Test Perps")

# %%
trainer.getPreplexity(data = validData, progressString = "Validation Perps")

# %%
inference = Inference(transformer, embeddings, data.getVocab())

# %%
perplexityScores = inference.getPerplexityScores(data = partitions['test'].getData()[:1000])

# %%
print(sum(perplexityScores)/len(perplexityScores))
