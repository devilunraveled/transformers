EMBEDDING_SIZE = 300

ANN_MODEL_SAMPLE_CONFIG = {
    'layers' : [
        {
            'in' : EMBEDDING_SIZE*5,
            'out' : 300
        },
        {
            'in' : 300,
            'out' : 35415
        }
    ],
    'activations' : [
        'relu',
    ]
}


LSTM_MODEL_SAMPLE_CONFIG = {
    'input_size' : EMBEDDING_SIZE,
    'hidden_size' : EMBEDDING_SIZE*2,
    'num_layers' : 1,
    'proj_size' : EMBEDDING_SIZE*2,
    'linear_layers' : [
        {
            'in' : EMBEDDING_SIZE*2,
            'out' : 35415
        }
    ],
    'activations' : [
    ]
}

DECODER_BLOCK_CONFIG = {
    'embedDim' : EMBEDDING_SIZE,
    'numHeads' : 6,
    'dropout' : 0.1,
    'batch_first' : True,
    'mlp_scaler' : 3,
}

TRANSFORMER_CONFIG = {
    'embedDim' : EMBEDDING_SIZE,
    'vocabSize' : 35415,
    'padToken' : 0,
    'numLayers' : 3,
    'blockConfig' : [
        DECODER_BLOCK_CONFIG,
        DECODER_BLOCK_CONFIG,
        DECODER_BLOCK_CONFIG
    ]
}


TRAINER_SAMPLE_CONFIG = {
    'optimizer' : 'Adam',
    'epochs' : 10,
    'learning_rate' : 5e-6
}
