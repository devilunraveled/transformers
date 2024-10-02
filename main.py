"""
Main code for training the transformer model.
"""

# Data preparation.
import pandas as pd
from config import CORPUS_PATH
from bidict import bidict
from config import SPECIAL_TOKENS
import pickle
from itertools import zip_longest

partitions = ['train', 'dev', 'test']
languages = ['en', 'fr']

filePaths = [f'{CORPUS_PATH}{partition}.{language}' for partition in partitions for language in languages]

englishSentences = {}
frenchSentences = {}

for file in filePaths :
    with open(file, 'r', encoding='utf-8') as f:
        parition, language = file.split('/')[-1].split('.')
        if language == 'en' :
            englishSentences[parition] = f.readlines()
        else :
            frenchSentences[parition] = f.readlines()

# Creating a dataset from these sentence.
data = []
for parition in partitions :
    for english, french in zip(englishSentences[parition], frenchSentences[parition]) :
        data.append({
            'id' : len(data) + 1,
            'english' : english,
            'french' : french,
            'partition' : parition
            })

data = pd.DataFrame(data)

print(data.head())

# Vocabularies.
englishVocab = bidict()
frenchVocab = bidict()

# %%
def createVocabulary(sentences, language):
    import nltk
    from collections import Counter
    
    tokens = [[SPECIAL_TOKENS['startToken']] + nltk.word_tokenize(text = sentence, language = language) + [SPECIAL_TOKENS['endToken']] for sentence in sentences]
    vocab = Counter(SPECIAL_TOKENS.values())
    
    print(tokens[0])
    print("Total tokens : ", sum(len(sentence) for sentence in tokens))

    for token in tokens :
        vocab.update(token)

    vocab = bidict((word, i) for i, word in zip(range(len(vocab)), vocab.keys()))
    
    print(len(vocab))
    
    return tokens, vocab

data['english_tokenized'], englishVocab = createVocabulary(data.english, 'english')
data['french_tokenized'], frenchVocab = createVocabulary(data.french, 'french')

with open('english_vocab.pkl', 'wb' ) as f :
    pickle.dump(englishVocab, f)

with open('french_vocab.pkl', 'wb' ) as f :
    pickle.dump(frenchVocab, f)

data['eng_data'] = data['english_tokenized'].apply( lambda x : [englishVocab[i] for i in x] )
data['fr_data'] = data['french_tokenized'].apply( lambda x : [frenchVocab[i] for i in x] )

def pad_sentences(eng_tokens, fr_tokens, pad_token=0):
    if len(eng_tokens) > len(fr_tokens) :
        fr_tokens = fr_tokens + [pad_token] * (len(eng_tokens) - len(fr_tokens))
    elif len(eng_tokens) < len(fr_tokens) :
        eng_tokens = eng_tokens + [pad_token] * (len(fr_tokens) - len(eng_tokens))
    return eng_tokens, fr_tokens

# Apply padding to each row
data[['padded_eng', 'padded_fr']] = data.apply( lambda row: pd.Series(pad_sentences(row['eng_data'], row['fr_data'])), axis=1 )

print(data['padded_eng'][4])
print(data['padded_fr'][4])

declutteredData = pd.DataFrame(data[['id', 'padded_eng', 'padded_fr', 'partition']])

with open('data.pkl', 'wb') as f:
    pickle.dump(declutteredData, f)
