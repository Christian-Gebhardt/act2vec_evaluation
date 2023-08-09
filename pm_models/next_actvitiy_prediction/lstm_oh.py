from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense

"""
Adjustable LSTM model (2 hidden layers) with one-hot-encoded activities
as input for next event prediction
"""
class LSTM_OH(Model):
    def __init__(self, layer_size, vocab_size, window_size):
        super(LSTM_OH, self).__init__()

        self.lstm1 = LSTM(layer_size[0], return_sequences=True, input_shape=(window_size, vocab_size))
        self.lstm2 = LSTM(layer_size[1])
        self.softmax = Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        output = self.softmax(x)
        return output
