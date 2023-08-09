from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense

"""
Adjustable LSTM model (2 hidden layers) with word vectors of activities
as inputs for next event prediction
"""
class LSTM_WV(Model):
    def __init__(self, layer_size, vocab_size, embedding_dim, word_vectors, window_size):
        super(LSTM_WV, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, weights=[word_vectors], input_length=window_size)
        self.lstm1 = LSTM(layer_size[0], return_sequences=True)
        self.lstm2 = LSTM(layer_size[1])
        self.softmax = Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        output = self.softmax(x)
        return output
