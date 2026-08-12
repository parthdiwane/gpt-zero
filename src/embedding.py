# code for embedding (intializaton phase i stil lneed to code the backrpop stuff lol)

import numpy as np

class embedding:
    def __init__(self, vocab_size, d_model, std = 0.02):
        self.weights = np.random.randn(vocab_size, d_model) * 0.02
    def get_embedding_weigths(self):
        return self.weights

    
