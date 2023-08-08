from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dense

"""
Adjustable FNN model (1 hidden layer) with word vectors of activities
as input for next event prediction
"""
class FNN_WV(Model):
    def __init__(self, layer_size, vocab_size, embedding_dim, word_vectors, window_length):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, weights=[word_vectors], input_length=window_length)
        self.flatten = Flatten()
        self.dense1 = Dense(layer_size[0], activation='relu')
        self.dense2 = Dense(layer_size[1], activation='relu')
        self.softmax = Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.softmax(x)
        return output
