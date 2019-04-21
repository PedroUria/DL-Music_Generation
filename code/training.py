import os
import music21 as ms  # python3 -m pip install --user music21 for installing on ubuntu instance
import numpy as np
import torch
import torch.nn as nn
from encoder_decoder import encode, decode
import matplotlib.pyplot as plt
from time import time
# NOTE: If you run this on the cloud (you should), you will need to download the files this code
# generates to evaluate them and compare with the original file/s.
# Use: scp -i ~/.ssh/my-ssh-key [USERNAME]@[IP_ADDRESS]:[REMOTE_FILE_PATH] [LOCAL_FILE_PATH] to do so

# ----------------------------------------------------
# Getting all the paths for the files under classical
# ----------------------------------------------------

path = os.getcwd()[:-4] + "data/classical"

files_by_author_and_subgenre = {}
# https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
for dire in [x[0] for x in os.walk(path)][1:]:
    if ".mid" in " ".join(os.listdir(dire)):
        files_by_author_and_subgenre[key][dire[dire.find(key) + len(key) + 1:]] = [dire + "/" + i for i in os.listdir(dire)]
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

# ----------------------------------------------------
# Reading and Encoding the right hand of a MIDI file
# ----------------------------------------------------


def get_right_hand(midi_file, time_step=0.05):
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

    return right_notes


# Encodes the first hand of bach
notes_encoded = get_right_hand(files_by_author["bach"][0], time_step=0.25)
notes_encoded = notes_encoded[:, :-1]  # Gets rid of the tempo dimension
# Reshapes to the suitable form to input on the LSTM
notes_encoded = torch.from_numpy(notes_encoded.reshape(-1, 1, 89)).float().cuda()
# print(notes_encoded.shape)  #  [seq_len, batch_size, input_size]

# -----------------------------------------
# Building our Neural Network Architecture
# -----------------------------------------


# Code modified from https://github.com/amir-jafari/Deep-Learning/blob/master/Pytorch_/7-RNN/1_RNN.py
class LSTMMusic(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMMusic, self).__init__()
        # Input of shape (seq_len, batch_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size)
        # Fully connected Layer at the end, output_size=input_size because we want to predict
        self.out = nn.Linear(hidden_size, input_size)  # the next note/sequence of notes
        # We use a sigmoid activation function instead of the usual softmax
        # because we want to predict potentially more than one label per vector,
        # like for example, when we have a hold or a chord
        # Idea from: https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
        self.act = nn.Sigmoid()

    def forward(self, x, h_c_state):
        y_pred, h_c_state = self.lstm(x, h_c_state)
        return self.act(self.out(y_pred)), h_c_state


# ----------------------------
# Training our Neural Network
# seq_len = 1: Use only the
# last note as input
# ----------------------------


def train_lstm_loss_whole_seq(seq_len, hidden_size=89, lr=0.01, n_epochs=100):

    """
    Training function where we compare all outputs to get the loss
    So if we input a sequence of 10 notes, we compare the predicted
    next note after each of the 10 notes with the real notes
    :param seq_len: Number of time steps to input as a sequence
    :param hidden_size: Number of neurons on the LSTM hidden layer
    :param lr: learning rate
    :return: class instance with learned parameters, loss per sample
    and loss per epoch
    """

    start = time()
    net = LSTMMusic(89, hidden_size).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.BCELoss()  # Because we are using sigmoid and not softmax, BCELoss is the right choice
    h_state = torch.zeros(1, 1, hidden_size).float().cuda()  #  Initializes the hidden
    c_state = torch.zeros(1, 1, hidden_size).float().cuda()  #  and cell states
    l = []  # Stores the loss per sequence
    lll = []  # Idem, but will be set to [] after each epoch for ll
    ll = []  #  Stores the mean loss per epoch
    wait_10 = 0  #  We will halve the learning rate if the loss does not decrease in the last 10 epochs
    for epoch in range(n_epochs):
        print("---------- epoch number:", epoch, "----------")
        for step in range(notes_encoded.shape[0] // seq_len - 1):
            x = notes_encoded[step:seq_len+step, :, :]  # We take as input the last seq_len notes
            x.requires_grad = True
            y = notes_encoded[step+1:seq_len+step+1, :, :]  # Uses the whole next sequence to get the loss
            y_pred, h_c_state = net(x, (h_state, c_state))
            # Repacks the hidden state, break the connection from last iteration
            h_state, c_state = h_c_state[0].data, h_c_state[1].data
            loss = loss_func(y_pred, y)
            l.append(loss.data)
            lll.append(loss.data)
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


def train_lstm_loss_only_last(seq_len, hidden_size=89, lr=0.01):

    """
    Training function where we compare only the last predicted note
    to get the loss, meaning that we want to focus on predicting the next note
    even if we input a sequence of many notes
    :param seq_len: Number of time steps to input as a sequence
    :param hidden_size: Number of neurons on the LSTM hidden layer
    :param lr: learning rate
    :return: class instance with learned parameters, loss per sample
    and loss per epoch
    """

    start = time()
    net = LSTMMusic(89, hidden_size).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.BCELoss()  # Because we are using sigmoid and not softmax, BCELoss is the right choice
    h_state = torch.zeros(1, 1, hidden_size).float().cuda()  #  Initializes the hidden
    c_state = torch.zeros(1, 1, hidden_size).float().cuda()  #  and cell states
    l = []  # Stores the loss per sequence
    lll = []  # Idem, but will be set to [] after each epoch for ll
    ll = []  #  Stores the mean loss per epoch
    wait_10 = 0  #  We will halve the learning rate if the loss does not decrease in the last 10 epochs
    for epoch in range(100):
        print("---------- epoch number:", epoch, "----------")
        for step in range(notes_encoded.shape[0] // seq_len - 1):
            x = notes_encoded[step:seq_len+step, :, :]  # We take as input the last seq_len notes
            x.requires_grad = True
            y = notes_encoded[seq_len+step:seq_len+step+1, :, :]  # Uses only the next note after input sequence to get the loss
            y_pred, h_c_state = net(x, (h_state, c_state))
            y_pred = y_pred[-1].reshape(1, 1, 89)  # Uses only the next note after input sequence to get the loss
            h_state, c_state = h_c_state[0].data, h_c_state[1].data  # repack the hidden state, break the connection from last iteration
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


# net, l, ll = train_lstm_loss_only_last(1)  # This takes 499.16 seconds on CPU
#        Last printed outputs:
# Halving learning rate from 0.0003125 to 0.00015625
# ---------- epoch number: 93 ----------
#            loss: 0.0027847283
# ---------- epoch number: 94 ----------
#            loss: 0.0031550413
# ---------- epoch number: 95 ----------
#            loss: 0.0027523746
# ---------- epoch number: 96 ----------
#            loss: 0.0025046463
# ---------- epoch number: 97 ----------
#            loss: 0.0023578731
# ---------- epoch number: 98 ----------
#            loss: 0.0023282936
# ---------- epoch number: 99 ----------
#            loss: 0.0021645506

# The training process took 258.39 seconds


def plot_loss(l, ll):

    """ Plots the loss per sample and per epoch"""

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


# plot_loss(l, ll)


# ------------------------------------------
# Using our Neural Network to generate music
# ------------------------------------------


def ltsm_gen(net, seq_len, file_name, n_steps=100, hidden_size=89, time_step=0.05):

    """ Uses the trained net to generate new notes
     and saves the output to a MIDI file """

    notes = []  # Will contain a sequence of the predicted notes
    x = torch.zeros((seq_len, 1, 89)).float().cuda()  # Input notes. All 0s minus the last one
    x[-1] = notes_encoded[0, :, :]  # which is the first note of the piece
    notes.append(x[-1].cpu().numpy())  # Saves the first note
    h_state = torch.zeros(1, 1, hidden_size).float().cuda()
    c_state = torch.zeros(1, 1, hidden_size).float().cuda()
    print_first = True
    for _ in range(n_steps):
        chosen = False  # To account for when no dimension's probability is bigger than 0.9
        y_pred, h_c_state = net(x, (h_state, c_state))
        h_state, c_state = h_c_state[0].data, h_c_state[1].data
        y_pred = y_pred.data
        y_pred = y_pred[-1]  # We only care about the last predicted note (next note after last note of input sequence)
        choose = torch.zeros((1, 1, 89))  # Coverts the probabilities to the actual note vector
        for idx in range(89):
            if y_pred[:, idx] > 0.9:
                choose[:, :, idx] = 1
                chosen = True
        if not chosen:
            if print_first:
                print("\nPrinting out the maximum prob of all notes for a time step",
                      "when this maximum prob is less than 0.9")
                print_first = False
            choose[:, :, np.argmax(y_pred.cpu())] = 1
            print(_, y_pred[:, np.argmax(y_pred.cpu())])  # Maximum probability out of all components
        x_new = torch.empty(x.shape)  # Uses the output of the last time_step
        for idx, nt in enumerate(x[1:]):  # As the input for the next time_step
            x_new[idx] = nt  # So the new sequence will be the same past sequence minus the first note
        x_new[-1] = choose  # of such past sequence, and plus the predicted note from this iteration
        x = x_new.cuda()  # We will use this new sequence to predict in the next iteration the next note
        notes.append(choose)  # Saves the predicted note

    # Gets the notes into the correct NumPy array shape
    gen_notes = np.empty((len(notes)-seq_len+1, 89))  # Doesn't use the first predicted notes
    for idx, nt in enumerate(notes[seq_len-1:]):  # Because of the 0s init problem  # TODO: Figure this out
        gen_notes[idx] = nt[0]

    # Decodes the generated music and saves it as a MIDI file
    gen_midi = decode(gen_notes, time_step=time_step)
    gen_midi.write("midi", file_name + ".mid")


# ltsm_gen(net, 1, "come_on")
# Printing out the maximum prob of all notes for a time step when this maximum prob is less than 0.9
# 1 tensor([0.2169], device='cuda:0')
# 2 tensor([0.0828], device='cuda:0')
# 9 tensor([0.4285], device='cuda:0')
# 10 tensor([0.8600], device='cuda:0')
# 17 tensor([0.1805], device='cuda:0')
# 18 tensor([0.2206], device='cuda:0')
# 21 tensor([0.7047], device='cuda:0')
# 25 tensor([0.6625], device='cuda:0')
# 26 tensor([0.6878], device='cuda:0')
# 33 tensor([0.2738], device='cuda:0')
# 35 tensor([0.4348], device='cuda:0')
# 38 tensor([0.7675], device='cuda:0')
# 41 tensor([0.6962], device='cuda:0')
# 43 tensor([0.4300], device='cuda:0')
# 49 tensor([0.5875], device='cuda:0')
# 51 tensor([0.7161], device='cuda:0')
# 57 tensor([0.4932], device='cuda:0')
# 58 tensor([0.6897], device='cuda:0')
# 59 tensor([0.5605], device='cuda:0')
# 60 tensor([0.6830], device='cuda:0')
# 62 tensor([0.8208], device='cuda:0')
# 67 tensor([0.6909], device='cuda:0')
# 75 tensor([0.4875], device='cuda:0')
# 83 tensor([0.6438], device='cuda:0')
# 91 tensor([0.5961], device='cuda:0')
# 99 tensor([0.6435], device='cuda:0')

# This is a decent result!

# ------------------------------------------
# Let's do the same with a seq_len = 14
# ------------------------------------------

# print("\nNow let's use sequences of 14 time steps as inputs",
#       "to train the network\n")
# net, l, ll = train_lstm_loss_only_last(14)

#        Last printed outputs:
# Halving learning rate from 0.005 to 0.0025
# ---------- epoch number: 72 ----------
#            loss: 0.0017278532
#                  .
#                  .
#                  .
# ---------- epoch number: 99 ----------
#            loss: 0.0010551634

# The training process took 23.44 seconds

# plot_loss(l, ll)
# ltsm_gen(net, 14, "come_on_14_14")
# Gets the first sequence but becomes stuck

# ------------------------------------------
# Let's do the same with a seq_len = 28
# ------------------------------------------

# print("\nUsing a 1 seq length for training and 28 for generating\n")
# net, l, ll = train_lstm_loss_only_last(1)
#        Last printed outputs:
# Halving learning rate from 0.0003125 to 0.00015625
# ---------- epoch number: 92 ----------
#            loss: 0.0029007166
# ---------- epoch number: 93 ----------
#            loss: 0.0023242815
# ---------- epoch number: 94 ----------
#            loss: 0.0021951161
# ---------- epoch number: 95 ----------
#            loss: 0.0025192962
# ---------- epoch number: 96 ----------
#            loss: 0.002035328
# ---------- epoch number: 97 ----------
#            loss: 0.0025752336
# ---------- epoch number: 98 ----------
#            loss: 0.0019571995
# ---------- epoch number: 99 ----------
#            loss: 0.0021617003
#
# The training process took 260.6 seconds

# plot_loss(l, ll)
# ltsm_gen(net, 28, "come_on_28")

# Printing out the maximum prob of all notes for a time step when this maximum prob is less than 0.9
# 9 tensor([0.4616], device='cuda:0')
# 17 tensor([0.3383], device='cuda:0')
# 18 tensor([0.4841], device='cuda:0')
# 25 tensor([0.7351], device='cuda:0')
# 33 tensor([0.1953], device='cuda:0')
# 41 tensor([0.8494], device='cuda:0')
# 49 tensor([0.3146], device='cuda:0')
# 57 tensor([0.6945], device='cuda:0')
# 58 tensor([0.5202], device='cuda:0')
# 59 tensor([0.7817], device='cuda:0')
# 65 tensor([0.8154], device='cuda:0')
# 66 tensor([0.4089], device='cuda:0')
# 73 tensor([0.4604], device='cuda:0')
# 81 tensor([0.2536], device='cuda:0')
# 82 tensor([0.4741], device='cuda:0')
# 89 tensor([0.4254], device='cuda:0')
# 90 tensor([0.8948], device='cuda:0')
# 97 tensor([0.5225], device='cuda:0')

# Much much better, check come_on_28.mid!


# net, l, ll = train_lstm_loss_only_last(28)
# Halving learning rate from 0.01 to 0.005
# ---------- epoch number: 43 ----------
#            loss: 0.008907785
#                   .
#                   .
#                   .
# ---------- epoch number: 99 ----------
#            loss: 0.0020988078
#
# The training process took 14.13 seconds
# plot_loss(l, ll)
# ltsm_gen(net, 28, "come_on_28_28")
# Printing out the maximum prob of all notes for a time step when this maximum prob is less than 0.9
# 2 tensor([0.6192], device='cuda:0')
# 9 tensor([0.5222], device='cuda:0')
# 17 tensor([0.5232], device='cuda:0')
# 25 tensor([0.5232], device='cuda:0')
# 33 tensor([0.5232], device='cuda:0')
# 41 tensor([0.5232], device='cuda:0')
# 49 tensor([0.5232], device='cuda:0')
# 57 tensor([0.5232], device='cuda:0')
# 65 tensor([0.5232], device='cuda:0')
# 73 tensor([0.5232], device='cuda:0')
# 81 tensor([0.5232], device='cuda:0')
# 89 tensor([0.5232], device='cuda:0')
# 97 tensor([0.5232], device='cuda:0')

# Again, stuck...


def ltsm_gen_v2(net, seq_len, file_name, n_steps=100, hidden_size=89, time_step=0.05):

    """ Uses the trained net to generate new notes
     and saves the output to a MIDI file """

    notes = []  # Will contain a sequence of the predicted notes
    x = notes_encoded[:seq_len]
    for nt in x:
        notes.append(nt.cpu().numpy())
    h_state = torch.zeros(1, 1, hidden_size).float().cuda()
    c_state = torch.zeros(1, 1, hidden_size).float().cuda()
    print_first = True
    for _ in range(n_steps):
        chosen = False  # To account for when no dimension's probability is bigger than 0.9
        y_pred, h_c_state = net(x, (h_state, c_state))
        h_state, c_state = h_c_state[0].data, h_c_state[1].data
        y_pred = y_pred.data
        y_pred = y_pred[-1]  # We only care about the last predicted note (next note after last note of input sequence)
        choose = torch.zeros((1, 1, 89))  # Coverts the probabilities to the actual note vector
        for idx in range(89):
            if y_pred[:, idx] > 0.9:
                choose[:, :, idx] = 1
                chosen = True
        if not chosen:
            if print_first:
                print("\nPrinting out the maximum prob of all notes for a time step",
                      "when this maximum prob is less than 0.9")
                print_first = False
            choose[:, :, np.argmax(y_pred.cpu())] = 1
            print(_, y_pred[:, np.argmax(y_pred.cpu())])  # Maximum probability out of all components
        x_new = torch.empty(x.shape)  # Uses the output of the last time_step
        for idx, nt in enumerate(x[1:]):  # As the input for the next time_step
            x_new[idx] = nt  # So the new sequence will be the same past sequence minus the first note
        x_new[-1] = choose  # of such past sequence, and plus the predicted note from this iteration
        x = x_new.cuda()  # We will use this new sequence to predict in the next iteration the next note
        notes.append(choose)  # Saves the predicted note

    # Gets the notes into the correct NumPy array shape
    gen_notes = np.empty((len(notes), 89))
    for idx, nt in enumerate(notes):
        gen_notes[idx] = nt[0]

    # Decodes the generated music and saves it as a MIDI file
    gen_midi = decode(gen_notes, time_step=time_step)
    gen_midi.write("midi", file_name + ".mid")


# net, l, ll = train_lstm_loss_whole_seq(28)
# plot_loss(l, ll)
# ltsm_gen_v2(net, 28, "come_on_28_1", n_steps=500)
# Good result
# ltsm_gen(net, 28, "come_on_28_1", n_steps=500)
# Also very good result, only using 1 note!!!

# TODO: Mini-batching, and maybe get


# Encodes the first hand of bach
notes_encoded = get_right_hand(files_by_author["beethoven"][-1], time_step=0.1)
notes_encoded = notes_encoded[:, :-1]  # Gets rid of the tempo dimension
# Reshapes to the suitable form to input on the LSTM
notes_encoded = torch.from_numpy(notes_encoded.reshape(-1, 1, 89)).float().cuda()

net, l, ll = train_lstm_loss_whole_seq(100, n_epochs=200)
#plot_loss(l, ll)
ltsm_gen_v2(net, 100, "elise_28_28", n_steps=500, time_step=0.1)



# print(files_by_author_and_subgenre["chopin"]["prelude"]