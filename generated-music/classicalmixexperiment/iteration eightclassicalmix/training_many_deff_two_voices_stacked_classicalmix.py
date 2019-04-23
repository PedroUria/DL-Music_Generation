# ------------------------------------------------------------
# This script consists on our approach to training the network
# on many songs by stacking them together sequentially
# ------------------------------------------------------------

import os
import music21 as ms  # python3 -m pip install --user music21 for installing on ubuntu instance
import numpy as np
import torch
import torch.nn as nn
from encoder_decoder import encode, decode
from combine import combine
import matplotlib.pyplot as plt
from time import time
from random import randint


# ----------------------------------------------------
# Getting all the paths for the files under classical
# ----------------------------------------------------


path = os.getcwd()[:-4] + "data/classical"

files_by_author_and_subgenre = {}
# https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
for dire in [x[0] for x in os.walk(path)][1:]:
    if ".mid" in " ".join(os.listdir(dire)):
        files_by_author_and_subgenre[key][dire[dire.find(key) + len(key) + 1:]] = [
            dire + "/" + i for i in os.listdir(dire)]
    else:
        key = dire[dire.find("classical/")+10:]
        files_by_author_and_subgenre[key] = {}
        
files_by_author = {}
for author, files in files_by_author_and_subgenre.items():
    files_by_author[author] = []
    for subgenre_files in files.values():   
        files_by_author[author] += subgenre_files
        
files_by_subgenre = {}
for files in files_by_author_and_subgenre.values():
    for key, filess in files.items():
        if key in files_by_subgenre:
            files_by_subgenre[key] += filess
        else:
            files_by_subgenre[key] = filess


# ------------------------------
# Defining our Loading Functions
# ------------------------------


def get_both_hands(midi_file, time_step=0.05):

    """
    Encodes the two hands of a MIDI file and
    Stacks them together horizontally
    Components [0:89] will be left hand
    And components [89:] will be right hand
    :param midi_file: path to the file
    :param time_step: Duration of each vector
    :return: Encoded matrix with both hands on it
    """

    # Reads the file and encodes each hand separately
    hands = ms.converter.parse(midi_file)
    voice = False  # If there is more than one voice on
    for idx, nt in enumerate(hands[0]):  # the right hand (first part), just
        if type(nt) == ms.stream.Voice:  # takes the first voice
            voice = True
            break
    if voice:
        right_notes = encode(hands[0][idx], time_step=time_step)
    else:
        right_notes = encode(hands[0], time_step=time_step)
    for idx, nt in enumerate(hands[1]):  # the left hand (second part), just
        if type(nt) == ms.stream.Voice:  # takes the first voice
            voice = True
            break
    if voice:
        left_notes = encode(hands[1][idx], time_step=time_step)
    else:
        left_notes = encode(hands[1], time_step=time_step)

    # Gets rid of the tempo component, we decided not to input it to our network
    right_notes, left_notes = right_notes[:, :-1], left_notes[:, :-1]

    # Stacks both hands together
    both = np.empty((max([right_notes.shape[0], left_notes.shape[0]]), 178))
    left, right = False, False  # We create a sequence of length = max length
    rest_shortest = np.zeros(89)  # between the two hands, and then encode the
    rest_shortest[87] = 1  # rest of the shortest hand as rests
    if left_notes.shape[0] > right_notes.shape[0]:
        longest = np.copy(left_notes)
        left = True
    elif right_notes.shape[0] > left_notes.shape[0]:
        longest = np.copy(right_notes)
        right = True
    for idx in range(both.shape[0]):
        try:
            both[idx, :] = np.hstack((left_notes[idx, :], right_notes[idx, :]))
        except IndexError:
            if left:
                both[idx, :] = np.hstack((longest[idx, :], rest_shortest))
            if right:
                both[idx, :] = np.hstack((rest_shortest, longest[idx, :]))

    return both


def load(author, subgenre, number, time_step=0.25):
    """
    Loads the given musical pieces
    :param author: Author's name
    :param subgenre: Sub-Genre
    :param number: Number of pieces to load
    :param time_step: Duration of each vector
    :return: List containing the encoded files
    """

    start = time()
    songs = files_by_author_and_subgenre[author][subgenre][:number]
    encoded_notes = []

    for i in range(len(songs)):
        try:
            notes = get_both_hands(songs[i], time_step=time_step)  # Encodes both hands of the piece
            encoded_notes.append(torch.from_numpy(notes.reshape(-1, 1, 178)).float().cuda())  # as tensor
            print("File number", i, "loaded")
        except:  # Our encoder appears to hae some bugs, probably related to the time step
            print("There was an error encoding this file", songs[i])
    print("The loading process took", round(time() - start), "seconds")

    return encoded_notes


# -----------------------------------------
# Building our Neural Network Architecture
# -----------------------------------------


# Code modified from https://github.com/amir-jafari/Deep-Learning/blob/master/Pytorch_/7-RNN/1_RNN.py
class LSTMMusic(nn.Module):

    """
    LSTM network that will try to learn the pattern within a series
    of musical pieces. It consists on a single LSTM layer followed
    by a fully connected Output Layer with a Sigmoid activation function
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(LSTMMusic, self).__init__()
        # Input of shape (seq_len, batch_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        # Fully connected Layer at the end, output_size=input_size because we want to predict
        self.out = nn.Linear(hidden_size, input_size)  # the next note/sequence of notes
        # We use a Sigmoid activation function instead of the usual Softmax
        # because we want to predict potentially more than one label per vector,
        # like for example, when we have a hold or a chord
        # Idea from: https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
        self.act = nn.Sigmoid()

    def forward(self, x, h_c_state):
        y_pred, h_c_state = self.lstm(x, h_c_state)
        return self.act(self.out(y_pred)), h_c_state


# --------------------------------
# Defining our Training Functions
# --------------------------------


def train_lstm_loss_only_last(seq_len, hidden_size=178, num_layers=1, dropout=0,
                              lr=0.01, n_epochs=100, use_all_seq=False, use_n_seq=100):

    """
    Training function where we compare only the last predicted note to get the loss,
    meaning that we want to focus on predicting the next note even if we input
    a sequence of many notes. If input_seq = [1, 2, 3], we get [2, 3, 4] as predicted
    but only use [4] and its real value to get the loss
    In this approach, we stack the sequences of each song together, sequentially
    And compute the loss and update the weights once for each sequence, on each epoch
    :param seq_len: Number of time steps to input as a sequence
    :param hidden_size: Number of neurons on the LSTM Layer. Default is number of input dims
    :param num_layers: Number of LSTM layers to stack together
    :param dropout: introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
    with dropout probability equal to dropout
    :param lr: Learning rate
    :param n_epochs: Number of training iterations
    :param use_all_seq: If True, uses all the sequences of the pieces. Default: False
    :param use_n_seq: Used when use_all_seq=False, we will only use these number of sequences (the first ones)
    :return: class instance with learned parameters, loss per sample and loss per epoch
    """

    start = time()

    # Stacks the first use_n_seq notes of each song together, sequentially
    if not use_all_seq:
        first_seq_notes_encoded = [notes[:use_n_seq * seq_len, :, :].cpu().numpy() for notes in notes_encoded]
        notes_encoded_stacked = first_seq_notes_encoded[0]
        for notes in first_seq_notes_encoded[1:]:
            notes_encoded_stacked = np.vstack((notes_encoded_stacked, notes))
        notes_encoded_stacked = torch.from_numpy(notes_encoded_stacked).cuda()
    # Stacks all songs together, sequentially
    else:
        first_seq_notes_encoded = [notes.cpu().numpy() for notes in notes_encoded]
        notes_encoded_stacked = first_seq_notes_encoded[0]
        for notes in first_seq_notes_encoded[1:]:
            notes_encoded_stacked = np.vstack((notes_encoded_stacked, notes))
            notes_encoded_stacked = torch.from_numpy(notes_encoded_stacked).cuda()

    net = LSTMMusic(178, hidden_size, num_layers, dropout).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.BCELoss()  # Because we are using sigmoid and not softmax, BCELoss is the right choice
    h_state = torch.zeros(num_layers, 1, hidden_size).float().cuda()  #  Initializes the hidden
    c_state = torch.zeros(num_layers, 1, hidden_size).float().cuda()  #  and cell states
    l = []  # Stores the loss per sequence
    lll = []  # Idem, but will be set to [] after each epoch for ll
    ll = []  #  Stores the mean loss per epoch
    wait_10 = 0  #  We will halve the learning rate if the loss does not decrease in the last 10 epochs
    n_seq = len(notes_encoded) * notes_encoded_stacked.shape[0] // seq_len  # We will use this number of sequences
    for epoch in range(n_epochs):
        print("---------- epoch number:", epoch, "----------")
        for step in range(n_seq):
            x = notes_encoded_stacked[step:seq_len+step, :, :]
            x.requires_grad = True
            # Uses only the next note after input sequence to get the loss
            y = notes_encoded_stacked[seq_len+step:seq_len+step+1, :, :]
            y_pred, h_c_state = net(x, (h_state, c_state))
            y_pred = y_pred[-1].reshape(1, 1, 178)  # Uses only the next note after input sequence to get the loss
            # repack the hidden state, break the connection from last iteration
            h_state, c_state = h_c_state[0].data, h_c_state[1].data
            loss = loss_func(y_pred, y)
            l.append(loss.data)
            lll.append(loss.data.cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ll.append(np.mean(lll))
        print("           loss:", ll[-1])
        if ll[-1] > np.mean(ll[::-1][:10]) and wait_10 >= 10:  #  We decrease the learning rate by half
            print("Halving learning rate from", lr, "to", lr / 2)  # When the loss stops decreasing
            lr = lr / 2
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            wait_10 = 0
        lll = []
        wait_10 += 1
    print("\nThe training process took", round(time() - start, 2), "seconds")
    return net, l, ll


def train_lstm_loss_whole_seq(seq_len, hidden_size=178, num_layers=1, dropout=0,
                              lr=0.01, n_epochs=100, use_all_seq=False, use_n_seq=100):

    """
    Training function where we compare all the notes predicted by the network for a given
    input sequence to get the loss. If input_seq = [1, 2, 3], we get [2, 3, 4] as predicted
    outputs and use all of them together with their true values to compute the loss.
    In this approach, we stack the sequences of each song together, sequentially
    And compute the loss and update the weights once for each sequence, on each epoch
    :param seq_len: Number of time steps to input as a sequence
    :param hidden_size: Number of neurons on the LSTM hidden layer
    :param num_layers: Number of LSTM layers to stack together
    :param dropout: introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
    with dropout probability equal to dropout
    :param lr: Learning rate
    :param n_epochs: Number of training iterations
    :param use_all_seq: If True, uses all the sequences of the pieces. Default: False
    :param use_n_seq: Used when use_all_seq=False, we will only use these number of sequences (the first ones)
    :return: class instance with learned parameters, loss per sample and loss per epoch
    """

    start = time()

    # Stacks the first use_n_seq notes of each song together, sequentially
    if not use_all_seq:
        first_seq_notes_encoded = [notes[:use_n_seq*seq_len, :, :].cpu().numpy() for notes in notes_encoded]
        notes_encoded_stacked = first_seq_notes_encoded[0]
        for notes in first_seq_notes_encoded[1:]:
            notes_encoded_stacked = np.vstack((notes_encoded_stacked, notes))
        notes_encoded_stacked = torch.from_numpy(notes_encoded_stacked).cuda()
    # Stacks all songs together, sequentially
    else:
        first_seq_notes_encoded = [notes.cpu().numpy() for notes in notes_encoded]
        notes_encoded_stacked = first_seq_notes_encoded[0]
        for notes in first_seq_notes_encoded[1:]:
            notes_encoded_stacked = np.vstack((notes_encoded_stacked, notes))
            notes_encoded_stacked = torch.from_numpy(notes_encoded_stacked).cuda()

    net = LSTMMusic(178, hidden_size, num_layers, dropout).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.BCELoss()  # Because we are using sigmoid and not softmax, BCELoss is the right choice
    h_state = torch.zeros(num_layers, 1, hidden_size).float().cuda()  #  Initializes the hidden
    c_state = torch.zeros(num_layers, 1, hidden_size).float().cuda()  #  and cell states
    l = []  # Stores the loss per sequence
    lll = []  # Idem, but will be set to [] after each epoch for ll
    ll = []  #  Stores the mean loss per epoch
    wait_10 = 0  #  We will halve the learning rate if the loss does not decrease in the last 10 epochs
    n_seq = len(notes_encoded)*notes_encoded_stacked.shape[0]//seq_len  # We will use this number of sequences
    for epoch in range(n_epochs):
        print("---------- epoch number:", epoch, "----------")
        for step in range(n_seq):
            x = notes_encoded_stacked[step:seq_len+step, :, :]
            x.requires_grad = True
            # Uses all the notes after input each note in input sequence to get the loss
            y = notes_encoded_stacked[step+1:seq_len+step+1, :, :]
            y_pred, h_c_state = net(x, (h_state, c_state))
            # Repacks the hidden state, break the connection from last iteration
            h_state, c_state = h_c_state[0].data, h_c_state[1].data
            # Computes the loss
            loss = loss_func(y_pred, y)
            l.append(loss.data)
            lll.append(loss.data.cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ll.append(np.mean(lll))
        print("           loss:", ll[-1])
        if ll[-1] > np.mean(ll[::-1][:10]) and wait_10 >= 10:  #  We decrease the learning rate by half
            print("Halving learning rate from", lr, "to", lr / 2)  # When the loss stops decreasing
            lr = lr / 2
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            wait_10 = 0
        lll = []
        wait_10 += 1
    print("\nThe training process took", round(time() - start, 2), "seconds")
    return net, l, ll


def plot_loss(l, ll):

    """ Plots the loss per sample and per epoch """

    plt.plot(range(len(l)), l)
    plt.title("Loss for each sample")
    plt.ylabel("Loss")
    plt.xlabel("Sample on each epoch")
    plt.show()

    plt.plot(range(len(ll)), ll)
    plt.title("Loss for each epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()


# ---------------------------------
# Defining our Generative Functions
# ---------------------------------


def get_tempo_dim_back(notes, tempo=74):

    """
    Adds an extra dimension for the tempo, this is needed
    because we took it out to train the network, but our
    encoder and decoder works includes the tempo
    :param notes: encoded matrix without the tempo dim
    :param tempo: value of the tempo to include
    :return: Same matrix with tempo dimension, in order
    to decode it successfully
    """

    c = np.empty((notes.shape[0], notes.shape[1]+1))
    for idx in range(notes.shape[0]):
        c[idx] = np.hstack((notes[idx], np.array([tempo])))
    return c


def ltsm_gen(net, seq_len, file_name, sampling_idx=0, sequence_start=0, n_steps=100, hidden_size=178,
             num_layers=1, time_step=0.05, changing_note=False, note_stuck=False, remove_extra_rests=False):

    """
    Uses the trained LSTM to generate new notes and saves the output to a MIDI file
    This approach uses a whole sequence of notes of one of the pieces we used to train
    the network, with length seq_len, which should be the same as the one used when training
    :param net: Trained LSTM
    :param seq_len: Length of input sequence
    :param file_name: Name to be given to the generated MIDI file
    :param sampling_idx: File to get the input sequence from, out of the pieces used to train the LSTM
    :param sequence_start: Index of the starting sequence, default to 0
    :param n_steps: Number of vectors to generate
    :param hidden_size: Hidden size of the trained LSTM
    :param num_layers: Number of layers of the trained LSTM
    :param time_step: Vector duration. Should be the same as the one on get_right_hand()
    :param changing_note: To sample from different sources at some point of the generation
    and add this new note to the sequence. This is done in case the generation gets stuck
    repeating a particular sequence over and over.
    :param note_stuck: To change the note if the generation gets stuck playing the same
    note over and over.
    :param remove_extra_rests: If the generation outputs a lot of rests in between, use this
    :return: None. Just saves the generated music as a .mid file
    """

    notes = []  # Will contain a sequence of the predicted notes
    x = notes_encoded[sampling_idx][sequence_start:sequence_start+seq_len]  # Uses the input sequence
    for nt in x:  # To start predicting. This will be later removed from
        notes.append(nt.cpu().numpy())  # the final output
    h_state = torch.zeros(num_layers, 1, hidden_size).float().cuda()
    c_state = torch.zeros(num_layers, 1, hidden_size).float().cuda()
    print_first = True  # To print out a message if every component of a
    # predicted vector is less than 0.9
    change_note = False

    for _ in range(n_steps):
        chosen = False  # To account for when no dimension's probability is bigger than 0.9
        y_pred, h_c_state = net(x, (h_state, c_state))  # Predicts the next notes for all
        h_state, c_state = h_c_state[0].data, h_c_state[1].data # the notes in the input sequence
        y_pred = y_pred.data  # We only care about the last predicted note
        y_pred = y_pred[-1]  # (next note after last note of input sequence)
        choose = torch.zeros((1, 1, 178))  # Coverts the probabilities to the actual note vector
        y_pred_left = y_pred[:, :89]
        for idx in range(89):
            if y_pred_left[:, idx] > 0.9:
                choose[:, :, idx] = 1
                chosen = True
        if y_pred_left[:, -1] >= 0.7:  # We add a hold condition, in case the probability
            choose[:, :, 88] = 1  # of having a hold is close to the one of having the pitch
        if not chosen:
            if print_first:
                print("\nPrinting out the maximum prob of all notes for a time step",
                      "when this maximum prob is less than 0.9")
                print_first = False
            pred_note_idx = np.argmax(y_pred_left.cpu())
            choose[:, :, pred_note_idx] = 1
            if pred_note_idx != 87:  # No holds for rests
                if y_pred_left[:, pred_note_idx] - y_pred_left[:, -1] <= 0.2:  # Hold condition
                    choose[:, :, 88] = 1
            print(_, "left", y_pred_left[:, np.argmax(y_pred_left.cpu())])  # Maximum probability out of all components
        y_pred_right = y_pred[:, 89:]
        for idx in range(89):
            if y_pred_right[:, idx] > 0.9:
                choose[:, :, idx + 89] = 1
                chosen = True
        if y_pred_right[:, -1] >= 0.7:
            choose[:, :, -1] = 1
        if not chosen:
            if print_first:
                print("\nPrinting out the maximum prob of all notes for a time step",
                      "when this maximum prob is less than 0.9")
                print_first = False
            pred_note_idx = np.argmax(y_pred_right.cpu())
            choose[:, :, pred_note_idx + 89] = 1
            if pred_note_idx != 87:  # No holds for rests
                if y_pred_right[:, pred_note_idx] - y_pred_right[:, -1] <= 0.2:  # Hold condition
                    choose[:, :, -1] = 1
            print(_, "right",
                  y_pred_right[:, np.argmax(y_pred_right.cpu())])  # Maximum probability out of all components
        x_new = torch.empty(x.shape)  # Uses the output of the last time_step
        for idx, nt in enumerate(x[1:]):  # As the input for the next time_step
            x_new[idx] = nt  # So the new sequence will be the same past sequence minus the first note
        x_new[-1] = choose
        x = x_new.cuda()  # We will use this new sequence to predict in the next iteration the next note
        notes.append(choose.cpu().numpy())  # Saves the predicted note

        # Condition so that the generation does not
        # get stuck on a particular sequence
        if changing_note:
            if _ % seq_len == 0:
                if sampling_idx >= len(notes_encoded):
                    sampling_idx = 0
                    change_note = True
                st = randint(1, 100)
                if change_note:
                    x_new[-1] = notes_encoded[sampling_idx][st, :, :]
                    change_note = False
                else:
                    x_new[-1] = notes_encoded[sampling_idx][0, :, :]
                sampling_idx += 1
                x = x_new.cuda()

        # Condition so that the generation does not
        # get stuck on a particular note
        if _ > 6 and note_stuck:
            if (notes[-1][:, :, 89:] == notes[-2][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                if (notes[-1][:, :, 89:] == notes[-3][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                    if (notes[-1][:, :, 89:] == notes[-4][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                        if (notes[-1][:, :, 89:] == notes[-5][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                            if (notes[-1][:, :, 89:] == notes[-6][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                                for m in range(5):
                                    notes.pop(-1)
                                if sampling_idx >= len(notes_encoded):
                                    sampling_idx = 0
                                x_new[-1] = notes_encoded[sampling_idx][randint(1, 100), :, :]
                                x = x_new.cuda()
                                sampling_idx += 1

    # Gets the notes into the correct NumPy array shape
    gen_notes = np.empty((len(notes) - seq_len + 1, 178))  # Doesn't use the first predicted notes
    for idx, nt in enumerate(notes[seq_len - 1:]):  # Because these were sampled from the training data
        gen_notes[idx] = nt[0]

    # Decodes the generated music
    gen_midi_left = decode(get_tempo_dim_back(gen_notes[:, :89], 74), time_step=time_step)
    # Gets rid of too many rests
    if remove_extra_rests:
        stream_left = ms.stream.Stream()
        for idx, nt in enumerate(gen_midi_left):
            if type(nt) == ms.note.Rest and idx < len(gen_midi_left) - 5:
                if nt.duration.quarterLength > 4 * time_step:
                    print("Removing rest")
                    continue
                if type(gen_midi_left[idx + 4]) == ms.note.Rest:
                    print("Removing rest")
                    continue
                stream_left.append(nt)
            else:
                stream_left.append(nt)
    else:
        stream_left = gen_midi_left
    # Same thing for right hand
    gen_midi_right = decode(get_tempo_dim_back(gen_notes[:, 89:], 74), time_step=time_step)
    if remove_extra_rests:
        stream_right = ms.stream.Stream()
        for idx, nt in enumerate(gen_midi_right):
            if type(nt) == ms.note.Rest and idx < len(gen_midi_right) - 5:
                if nt.duration.quarterLength > 4 * time_step:
                    print("Removing rest")
                    continue
                if type(gen_midi_right[idx + 4]) == ms.note.Rest:
                    print("Removing rest")
                    continue
                stream_right.append(nt)
            else:
                stream_right.append(nt)
    else:
        stream_right = gen_midi_right

    # Saves both hands combined as a MIDI file
    combine(stream_left, stream_right, file_name + ".mid")


def ltsm_gen_v2(net, seq_len, file_name, sampling_idx=0, note_pos=0, n_steps=100, hidden_size=178,
                num_layers=1, time_step=0.05, changing_note=False, note_stuck=False, remove_extra_rests=False):

    """
    Uses the trained LSTM to generate new notes and saves the output to a MIDI file
    The difference between this and the previous one is that we only use one note as input
    And then keep generating notes until we have a sequence of notes of length = seq_len
    Once we do, we start appending the generated notes to the final output
    :param net: Trained LSTM
    :param seq_len: Length of input sequence
    :param file_name: Name to be given to the generated MIDI file
    :param sampling_idx: File to get the input note from, out of the pieces used to train the LSTM
    :param note_pos: Position of the sampled input note in the source piece, default to the first note
    :param n_steps: Number of vectors to generate
    :param hidden_size: Hidden size of the trained LSTM
    :param num_layers: Number of layers of the trained LSTM
    :param time_step: Vector duration. Should be the same as the one on get_right_hand()
    :param changing_note: To sample from different sources at some point of the generation
    and add this new note to the sequence. This is done in case the generation gets stuck
    repeating a particular sequence over and over.
    :param note_stuck: To change the note if the generation gets stuck playing the same
    note over and over.
    :param remove_extra_rests: If the generation outputs a lot of rests in between, use this
    :return: None. Just saves the generated music as a .mid file
    """

    notes = []  # Will contain a sequence of the predicted notes
    x = notes_encoded[sampling_idx][note_pos:note_pos+1, :, :]  # First note of the piece
    notes.append(x.cpu().numpy())  # Saves the first note
    h_state = torch.zeros(num_layers, 1, hidden_size).float().cuda()
    c_state = torch.zeros(num_layers, 1, hidden_size).float().cuda()
    print_first = True
    change_note = False
    for _ in range(n_steps):
        chosen = False  # To account for when no dimension's probability is bigger than 0.9
        y_pred, h_c_state = net(x, (h_state, c_state))
        h_state, c_state = h_c_state[0].data, h_c_state[1].data
        y_pred = y_pred.data
        y_pred = y_pred[-1]  # We only care about the last predicted note (next note after last note of input sequence)
        choose = torch.zeros((1, 1, 178))  # Coverts the probabilities to the actual note vector
        y_pred_left = y_pred[:, :89]
        for idx in range(89):
            if y_pred_left[:, idx] > 0.9:
                choose[:, :, idx] = 1
                chosen = True
        if y_pred_left[:, -1] >= 0.7:  # We add a hold condition, in case the probability
            choose[:, :, 88] = 1  # of having a hold is close to the one of having the pitch
        if not chosen:
            if print_first:
                print("\nPrinting out the maximum prob of all notes for a time step",
                      "when this maximum prob is less than 0.9")
                print_first = False
            pred_note_idx = np.argmax(y_pred_left.cpu())
            choose[:, :, pred_note_idx] = 1
            if pred_note_idx != 87:  # No holds for rests
                if y_pred_left[:, pred_note_idx] - y_pred_left[:, -1] <= 0.2:  # Hold condition
                    choose[:, :, 88] = 1
            print(_, "left", y_pred_left[:, np.argmax(y_pred_left.cpu())])  # Maximum probability out of all components
        y_pred_right = y_pred[:, 89:]
        for idx in range(89):
            if y_pred_right[:, idx] > 0.9:
                choose[:, :, idx+89] = 1
                chosen = True
        if y_pred_right[:, -1] >= 0.7:
            choose[:, :, -1] = 1
        if not chosen:
            if print_first:
                print("\nPrinting out the maximum prob of all notes for a time step",
                      "when this maximum prob is less than 0.9")
                print_first = False
            pred_note_idx = np.argmax(y_pred_right.cpu())
            choose[:, :, pred_note_idx+89] = 1
            if pred_note_idx != 87:  # No holds for rests
                if y_pred_right[:, pred_note_idx] - y_pred_right[:, -1] <= 0.2:  # Hold condition
                    choose[:, :, -1] = 1
            # Maximum probability out of all components
            print(_, "right", y_pred_right[:, np.argmax(y_pred_right.cpu())])

        # If the number of input sequences is shorter than the expected one
        if x.shape[0] < seq_len:  # We keep adding the predicted notes to this input
            x_new = torch.empty((x.shape[0] + 1, x.shape[1], x.shape[2]))
            for i in range(x_new.shape[0] - 1):
                x_new[i, :, :] = x[i, :, :]
            x_new[-1, :, :] = y_pred
            x = x_new.cuda()
            notes.append(choose)
        else:  # If we already have enough sequences
            x_new = torch.empty(x.shape)  # Removes the first note
            for idx, nt in enumerate(x[1:]):  # of the current sequence
                x_new[idx] = nt  # And appends the predicted note to the
            x_new[-1] = choose  # input of sequences
            x = x_new.cuda()
            notes.append(choose)

        # Condition so that the generation does not
        # get stuck on a particular sequence
        if changing_note:
            if _ % seq_len == 0:
                if sampling_idx >= len(notes_encoded):
                    sampling_idx = 0
                    change_note = True
                st = randint(1, 100)
                if change_note:
                    x_new[-1] = notes_encoded[sampling_idx][st, :, :]
                    change_note = False
                else:
                    x_new[-1] = notes_encoded[sampling_idx][0, :, :]
                sampling_idx += 1
                x = x_new.cuda()

        # Condition so that the generation does not
        # get stuck on a particular note
        if _ > 8 and note_stuck:
            if (notes[-1][:, :, 89:] == notes[-2][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                if (notes[-1][:, :, 89:] == notes[-3][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                    if (notes[-1][:, :, 89:] == notes[-4][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                        if (notes[-1][:, :, 89:] == notes[-5][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                            if (notes[-1][:, :, 89:] == notes[-6][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                                for m in range(5):
                                    notes.pop(-1)
                                if sampling_idx >= len(notes_encoded):
                                    sampling_idx = 0
                                x_new[-1] = notes_encoded[sampling_idx][randint(1, 100), :, :]
                                x = x_new.cuda()
                                sampling_idx += 1

    # Gets the notes into the correct NumPy array shape
    gen_notes = np.empty((len(notes)-seq_len+1, 178))  # Doesn't use the first predicted notes
    for idx, nt in enumerate(notes[seq_len-1:]):  # Because at first this will be inaccurate
        gen_notes[idx] = nt[0]

    # Decodes the generated music
    gen_midi_left = decode(get_tempo_dim_back(gen_notes[:, :89], 74), time_step=time_step)
    # Gets rid of too many rests
    if remove_extra_rests:
        stream_left = ms.stream.Stream()
        for idx, nt in enumerate(gen_midi_left):
            if type(nt) == ms.note.Rest and idx < len(gen_midi_left) - 5:
                if nt.duration.quarterLength > 4*time_step:
                    print("Removing rest")
                    continue
                if type(gen_midi_left[idx + 4]) == ms.note.Rest:
                    print("Removing rest")
                    continue
                stream_left.append(nt)
            else:
                stream_left.append(nt)
    else:
        stream_left = gen_midi_left
    # Same thing for right hand
    gen_midi_right = decode(get_tempo_dim_back(gen_notes[:, 89:], 74), time_step=time_step)
    if remove_extra_rests:
        stream_right = ms.stream.Stream()
        for idx, nt in enumerate(gen_midi_right):
            if type(nt) == ms.note.Rest and idx < len(gen_midi_right) - 5:
                if nt.duration.quarterLength > 4 * time_step:
                    print("Removing rest")
                    continue
                if type(gen_midi_right[idx + 4]) == ms.note.Rest:
                    print("Removing rest")
                    continue
                stream_right.append(nt)
            else:
                stream_right.append(nt)
    else:
        stream_right = gen_midi_right

    # Saves both hands combined as a MIDI file
    combine(stream_left, stream_right, file_name + ".mid")


# -------------
# Some Attempts
# -------------

notes_encoded = load("classicalmix", "multipleclassical", 4, time_step=0.2)
net, l, ll = train_lstm_loss_whole_seq(100, n_epochs=100, lr=0.01)
torch.save(net.state_dict(), 'classicalmix_it8_both_stacked_1.pkl')
# net = LSTMMusic(178, 178).cuda()
# net.load_state_dict(torch.load("beethoven_both_stacked_1.pkl"))
# net.eval()
ltsm_gen_v2(net, 100, "classicalmix_it8_both_stacked_1", sampling_idx=0, time_step=0.2, n_steps=1000)


# notes_encoded = load("mendelssohn", "romantic", 10)
# net, l, ll = train_lstm_loss_whole_seq(50, n_epochs=100, lr=0.01)
# torch.save(net.state_dict(), 'lstm_whole_seq_mendelssohn_both_stacked.pkl')
# net = LSTMMusic(178, 178).cuda()
# net.load_state_dict(torch.load("lstm_whole_seq_mendelssohn_both_stacked.pkl"))
# net.eval()
# ltsm_gen_v2(net, 50, "mendelssohn_both_stacked_3", time_step=0.25, n_steps=1000, note_stuck=True)
# Decent!

# notes_encoded = load("mendelssohn", "romantic", 10)
# net, l, ll = train_lstm_loss_whole_seq(50, n_epochs=100, lr=0.01)
# torch.save(net.state_dict(), 'lstm_whole_seq_mendelssohn_both_stacked_1.pkl')
# net = LSTMMusic(178, 178).cuda()
# net.load_state_dict(torch.load("lstm_whole_seq_mendelssohn_both_stacked_1.pkl"))
# net.eval()
# ltsm_gen_v2(net, 50, "mendelssohn_both_stacked_1", time_step=0.25, n_steps=1000)
# Pretty good!!

# notes_encoded = load("mendelssohn", "romantic", 10)
# net, l, ll = train_lstm_loss_whole_seq(50, n_epochs=100, lr=0.01)
# torch.save(net.state_dict(), 'lstm_whole_seq_mendelssohn_both_stacked_2.pkl')
# net = LSTMMusic(178, 178).cuda()
# net.load_state_dict(torch.load("lstm_whole_seq_mendelssohn_both_stacked_2.pkl"))
# net.eval()
# ltsm_gen_v2(net, 50, "mendelssohn_both_stacked_2", time_step=0.25, n_steps=1000, note_stuck=True)
# Not bad!
