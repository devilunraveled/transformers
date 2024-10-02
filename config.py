CORPUS_PATH = './corpus/ted-talks-corpus/'

SPECIAL_TOKENS = {
    'padToken' : '<pad>',
    'startToken' : '<start>',
    'endToken' : '<end>'
}

ENCODER_CONFIG = {
    'inputDim' : 512,
    'numHeads' : 8,
    'dropout'  : 0.1,
    'mlp_scaler' : 2,
    'outputDim' : 512,
}

DECODER_CONFIG = {
    'inputDim' : 512,
    'numHeads' : 8,
    'dropout'  : 0.1,
    'mlp_scaler' : 2,
    'outputDim' : 512,
}

MODEL_CONFIG = {
    'num_enc_embeddings' : 27121,
    'num_dec_embeddings' : 36420,
    'inputDim' : 512,
    'outputDim' : 512,
    'padding_idx' : 0,
    'numEncoderLayers' : 4,
    'numDecoderLayers' : 4,
    'modelOutputDimension' : 512,
    'outputVocabSize' : 36420,
    'encoderBlockConfig' : [ENCODER_CONFIG for _ in range(4)],
    'decoderBlockConfig' : [DECODER_CONFIG for _ in range(4)]
}
