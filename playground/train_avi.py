import glob
import pickle
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

# Prepare the input and output data from the left/right hand seaquences
def prepare(data):
    """
    Takes the dataset and shifts data to produce the input and target
    """
    data_input = data[:-1]
    data_output = data[1:]
    return data_input, data_output

# Prepare the model
def create_network(network_input, n_vocab):
    """
    Create the structure of the neural network
    """
    model = Sequential()
    model.add(LSTM(
        75,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(75, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(80))
    model.add(Dense(85))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    """ 
    Train the neural network
    """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

def train_network(data):
    """ 
    Train a Neural Network to generate music
    """
    network_input, network_output = prepare(data)
    model = create_network(network_input, left.shape[1])
    train(model, network_input, network_output)

def read(filepath):
    return np.genfromtxt(filepath, delimiter=',')

if __name__ == "__main__":
    data = read('left.csv')
    train_network(data)
    