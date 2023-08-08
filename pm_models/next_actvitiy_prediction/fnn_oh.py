from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense


"""
Adjustable FNN model (1 hidden layer) with one-hot-encoded activities
as input for next event prediction
"""
class FNN_OH(Model):
    def __init__(self, layer_size, vocab_size, window_length):
        super().__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(layer_size[0], activation='relu', input_shape=(window_length, vocab_size))
        self.dense2 = Dense(layer_size[1], activation='relu')
        self.softmax = Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        output = self.softmax(x)
        return output
