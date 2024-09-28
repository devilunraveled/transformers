DataConfig : dict = {
    # File Structure
    'train' : {
        'name' : 'train',
        'path' : 'data/train.txt'
    },
    'test' : {
        'name' : 'test',
        'path' : 'data/test.txt'
    },
    'valid' : {
        'name' : 'valid',
        'path' : 'data/valid.txt'
    },

    # Utility Constants
    'regex' : r'[^A-Za-z0-9 ]+',
    'customTokens' : {
        'start'     : '<SOS>',
        'end'       : '<EOS>',
        'pad'       : '<PAD>',
        'unknown'   : '<UNK>'
    },
    'padding' : 5
}
