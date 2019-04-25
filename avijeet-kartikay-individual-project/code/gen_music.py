import glob
import pickle
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation

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
    model.add(LSTM(89, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(80))
    model.add(Dense(85))
    model.add(Dropout(0.5))
    model.add(Dense(n_vocab))
    model.add(Activation('relu'))
    adam = optimizers.Adam(lr=0.0001, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=adam)

    return model

def generator(model, x):
    notes = []
    seq = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    for cnt in range(30):
        note = []
#	seq = np.reshape(x, (x.shape[0], 1, x.shape[1]))
	y_pred = model.predict()
        for val in y_pred[0]:
	    for each in val:
                if val >= 0.55:
                    note.append(1)
            	else:
                    note.append(0)
            x = np.array(note)
        notes.append(x)
	seq = np.reshape(notes[-10:], (10, 1, notes[-1].shape[0]))
    return notes

def generate_notes(input_data, model_path):
    data_input = np.genfromtxt(input_data,delimiter=',')
    data = np.reshape(data_input, (data_input.shape[0], 1, data_input.shape[1]))
#    data_input = np.genfromtxt(input_data,delimiter=',')
    sel = np.random.randint(10) * np.random.randint(10)
    x = data_input[sel:sel+10]
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    model = create_network(data, data_input.shape[1])
    model.load_weights(model_path)

    return generator(model, x)

def savefile(op_list, filename):
    data = []
    for each in op_list:
        data.append(np.array(each))
    data = np.array(data)
    np.savetxt(filename, data, delimiter=',')

left = generate_notes('left.csv', 'left.hdf5')
right = generate_notes('right.csv', 'right.hdf5')

savefile(left, 'left_gen.csv')
savefile(right, 'right_gen.csv')

