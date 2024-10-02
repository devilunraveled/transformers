"""
Contains the code for training the model.
"""
from torch.nn.utils.rnn import pad_sequence as PadSequence
from dataHandler import TranslationDataset
from torch.utils.data import DataLoader
from training.trainer import Trainer
from src.transformer import EncoderDecoderTransformer
from config import SPECIAL_TOKENS, MODEL_CONFIG
import pickle

import torch
# torch.autograd.set_detect_anomaly(True)

with open('./data.pkl', 'rb') as f:
    declutteredData = pickle.load(f)

with open('./french_vocab.pkl', 'rb' ) as f :
    frenchVocab = pickle.load(f)

with open('./english_vocab.pkl', 'rb' ) as f :
    englishVocab = pickle.load(f)

def dataCollater(batch, padToken):
    english, french = zip(*batch)
    english = PadSequence(english, padding_value = padToken, batch_first=True)
    french = PadSequence(french, padding_value = padToken, batch_first=True)
    
    return english, french

trainData = DataLoader(TranslationDataset(declutteredData[declutteredData.partition == 'train']), batch_size=4,shuffle=False, num_workers=4, collate_fn= lambda x : dataCollater(x, frenchVocab[SPECIAL_TOKENS['padToken']]))
devData = DataLoader(TranslationDataset(declutteredData[declutteredData.partition == 'dev']), batch_size=1, shuffle=False, num_workers=4, collate_fn= lambda x : dataCollater(x, frenchVocab[SPECIAL_TOKENS['padToken']]))
testData = DataLoader(TranslationDataset(declutteredData[declutteredData.partition == 'test']), batch_size=1, shuffle=False, num_workers=4, collate_fn= lambda x : dataCollater(x, frenchVocab[SPECIAL_TOKENS['padToken']]))

# Instantiating the model.
# model = EncoderDecoderTransformer.load('./ckpts/models/EncoderDecoderTransformer_10.pt')
model = EncoderDecoderTransformer(MODEL_CONFIG)

print(f"Model on {model.device}")

# Training.
trainer = Trainer(model = model, trainData = trainData, validData = devData, optimizer = 'Adam', learning_rate=1e-4)

trainer.train(10, save = True, progressString = 'Epoch')
print(trainer.test())

# Testing.
output, gold = trainer.getModelResponse(frenchVocab, data = testData)

with open('./output.pkl', 'wb') as f :
    pickle.dump(output, f)

with open('./gold.pkl', 'wb') as f :
    pickle.dump(gold, f)

print(output[19],"\n", gold[19])
