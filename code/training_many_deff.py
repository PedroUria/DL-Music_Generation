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

# ------------------------------
# Defining our Loading Functions
# ------------------------------


def get_right_hand(midi_file, time_step=0.05):

    """
    Gets the encoded right hand of a midi file
    :param midi_file: path to the file
    :param time_step: Duration of each vector
    :return:
    """

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


def load(author, subgenre, number):
    """
    Loads the given musical pieces
    :param author: Author's name
    :param subgenre: Genre
    :param number: Number of pieces to load
    :return: Dictionary containing the encoded files
    """

    start = time()
    songs = files_by_author_and_subgenre[author][subgenre][:number]
    encoded_notes = {}

    for i in range(len(songs)):
        encoded_notes[i] = get_right_hand(songs[i], time_step=0.25)  # Encodes the right hand of the piece
        encoded_notes[i] = encoded_notes[i][:, :-1]  # Gets rid of the tempo dimension and prepares the input
        encoded_notes[i] = torch.from_numpy(encoded_notes[i].reshape(-1, 1, 89)).float().cuda()  # as tensor
        print("File number", i, "loaded")
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

    def __init__(self, input_size, hidden_size):
        super(LSTMMusic, self).__init__()
        # Input of shape (seq_len, batch_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size)
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


def train_lstm_loss_only_last(seq_len, hidden_size=89, lr=0.01, n_epochs=100):

    """
    Training function where we compare only the last predicted note to get the loss,
    meaning that we want to focus on predicting the next note even if we input
    a sequence of many notes. If input_seq = [1, 2, 3], we get [2, 3, 4] as predicted
    but only use [4] and its real value to get the loss
    :param seq_len: Number of time steps to input as a sequence
    :param hidden_size: Number of neurons on the LSTM Layer. Default is number of input dims
    :param lr: Learning rate
    :param n_epochs: Number of training iterations
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
    len_piece = []
    # batch_size = len(songs)
    for nts in notes_encoded.values():
        len_piece.append(nts.shape[0])
    n_seq = min(len_piece) - seq_len - 1
    for epoch in range(n_epochs):
        print("---------- epoch number:", epoch, "----------")
        for step in range(n_seq):
            loss = 0
            for i in range(len(notes_encoded)):
                x = notes_encoded[i][step:seq_len+step, :, :]
                x.requires_grad = True
                # Uses only the next note after input sequence to get the loss
                y = notes_encoded[i][seq_len+step:seq_len+step+1, :, :]
                y_pred, h_c_state = net(x, (h_state, c_state))
                y_pred = y_pred[-1].reshape(1, 1, 89)  # Uses only the next note after input sequence to get the loss
                # repack the hidden state, break the connection from last iteration
                h_state, c_state = h_c_state[0].data, h_c_state[1].data
                loss += loss_func(y_pred, y)
            loss = loss/len(notes_encoded)
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


def train_lstm_loss_whole_seq(seq_len, hidden_size=89, lr=0.01, n_epochs=100, use_all_seq=False, use_n_seq=100):

    """
    Training function where we compare all the notes predicted by the network for a given
    input sequence to get the loss. If input_seq = [1, 2, 3], we get [2, 3, 4] as predicted
    outputs and use all of them together with their true values to compute the loss.
    :param seq_len: Number of time steps to input as a sequence
    :param hidden_size: Number of neurons on the LSTM hidden layer
    :param lr: Learning rate
    :param n_epochs: Number of training iterations
    :param use_all_seq: If True, uses all the sequences of the pieces. Default: False
    :param use_n_seq: Used when use_all_seq=False, we will only use these number of sequences (the first ones)
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
    len_piece = []
    for nts in notes_encoded.values():
        len_piece.append(nts.shape[0])
    if use_all_seq:
        n_seq = min(len_piece) - seq_len - 1
    else:
        n_seq = use_n_seq  # Uses only the first use_n_seq sequences of each piece
    for epoch in range(n_epochs):
        print("---------- epoch number:", epoch, "----------")
        for step in range(n_seq):
            loss = 0
            for i in range(len(notes_encoded)):
                x = notes_encoded[i][step:seq_len+step, :, :]
                x.requires_grad = True
                # Uses only the next note after input sequence to get the loss
                y = notes_encoded[i][step+1:seq_len+step+1, :, :]
                y_pred, h_c_state = net(x, (h_state, c_state))
                # Repacks the hidden state, break the connection from last iteration
                h_state, c_state = h_c_state[0].data, h_c_state[1].data
                loss += loss_func(y_pred, y)
            loss = loss/len(notes_encoded)
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


# ---------------------------------
# Defining our Generative Functions
# ---------------------------------


def ltsm_gen(net, seq_len, file_name, sampling_index=0, n_steps=100, hidden_size=89, time_step=0.05):

    """
    Uses the trained LSTM to generate new notes and saves the output to a MIDI file
    :param net: Trained LSTM
    :param seq_len: Length of input sequence
    :param file_name: Name to be given to the generated MIDI file
    :param sampling_index: File to get the input sequence from, out of the pieces used to train the LSTM
    :param n_steps: Number of vectors to generate
    :param hidden_size: Hidden size of the trained LSTM
    :param time_step: Vector duration. Should be the same as the one on get_right_hand()
    :return: None. Just saves the generated music as a .mid file
    """

    notes = []  # Will contain a sequence of the predicted notes
    x = notes_encoded[sampling_index][:seq_len]  # Uses the input sequence
    for nt in x:  # To start predicting. This will be later removed from
        notes.append(nt.cpu().numpy())  # the final output
    h_state = torch.zeros(1, 1, hidden_size).float().cuda()
    c_state = torch.zeros(1, 1, hidden_size).float().cuda()
    print_first = True  # To print out a message if every component of a
    # predicted vector is less than 0.9
    # change_note = False  # TODO: Use this to break the stucking loop

    for _ in range(n_steps):
        chosen = False  # To account for when no dimension's probability is bigger than 0.9
        y_pred, h_c_state = net(x, (h_state, c_state))  # Predicts the next notes for all
        h_state, c_state = h_c_state[0].data, h_c_state[1].data # the notes in the input sequence
        y_pred = y_pred.data  # We only care about the last predicted note
        y_pred = y_pred[-1]  # (next note after last note of input sequence)
        choose = torch.zeros((1, 1, 89))  # Coverts the probabilities to the actual note vector
        for idx in range(89):
            if y_pred[:, idx] > 0.9:
                choose[:, :, idx] = 1
                chosen = True
                if y_pred[:, -1] <= 0.7:  # We add a hold condition, in case the probability
                    choose[:, :, -1] = 1  # of having a hold is close to the one of having the pitch
        if not chosen:  # If no dimension is greater than 0.9
            if print_first:
                print("\nPrinting out the maximum prob of all notes for a time step",
                      "when this maximum prob is less than 0.9")
                print_first = False
            # Predicts the note as the one with the highest probability
            pred_note_idx = np.argmax(y_pred.cpu())
            choose[:, :, pred_note_idx] = 1
            if pred_note_idx != 88:  # We do not want to add holds for rests in this case
                if y_pred[:, pred_note_idx] - y_pred[:, -1] <= 0.2:  # Hold condition
                    choose[:, :, -1] = 1
            print(_, y_pred[:, np.argmax(y_pred.cpu())])  # Maximum probability out of all components
        x_new = torch.empty(x.shape)  # Uses the output of the last time_step
        for idx, nt in enumerate(x[1:]):  # As the input for the next time_step
            x_new[idx] = nt  # So the new sequence will be the same past sequence minus the first note

        # To avoid repeating sequences
        # repeat = False
        #         if len(notes) > seq_len+2:
        #             repeat = True
        #             for idxx, nt in enumerate(notes[::-1][:10]):
        #                 # print(type(nt), type(notes[::-1][10 + idxx]))
        #                 #print(notes[::-1][10+idxx].shape)
        #                 #print(10 + idxx, "---", len(notes), len(notes[::-1]))
        #                 try:
        #                     if (nt == notes[::-1][10 + idxx]).sum() != 89:
        #                         repeat = False
        #                 except:
        #                     pass
        #             for idxx, nt in enumerate(notes[::-1][:seq_len]):
        #                 try:
        #                     if (nt == notes[::-1][seq_len + idxx]).sum() != 89:
        #                         repeat = False
        #                 except:
        #                     pass
        #         if repeat:  # To avoid repeating sequences
        #             if sampling_idx >= len(notes_encoded):
        #                 sampling_idx = 0
        #                 change_note = True
        #             if change_note:
        #                 x_new[-1] = notes_encoded[sampling_idx][np.random.randint(1, 100, size=1)[0], :, :]
        #                 change_note = False
        #             else:
        #                 x_new[-1] = notes_encoded[sampling_idx][0, :, :]
        #             sampling_idx += 1
        #else:
            # x_new[-1] = choose  # of such past sequence, and plus the predicted note from this iteration

        x_new[-1] = choose
        x = x_new.cuda()  # We will use this new sequence to predict in the next iteration the next note
        notes.append(choose.cpu().numpy())  # Saves the predicted note

    # Gets the notes into the correct NumPy array shape
    gen_notes = np.empty((len(notes) - seq_len + 1, 89))  # Doesn't use the first predicted notes
    for idx, nt in enumerate(notes[seq_len - 1:]):  # Because these were sampled from the training data
        gen_notes[idx] = nt[0]

    # Decodes the generated music and saves it as a MIDI file
    gen_midi = decode(gen_notes, time_step=time_step)
    gen_midi.write("midi", file_name + ".mid")


def ltsm_gen_v2(net, seq_len, file_name, n_steps=100, hidden_size=89, time_step=0.05):

    """
    Uses the trained LSTM to generate new notes and saves the output to a MIDI file
    The difference between this and the previos one is that we only use one note as input
    And then keep generating notes until we have a sequence of notes of length = seq_len
    Once we do, we start appending the generated notes to the final output
    :param net: Trained LSTM
    :param seq_len: Length of input sequence
    :param file_name: Name to be given to the generated MIDI file
    :param n_steps: Number of vectors to generate
    :param hidden_size: Hidden size of the trained LSTM
    :param time_step: Vector duration. Should be the same as the one on get_right_hand()
    :return: None. Just saves the generated music as a .mid file
    """

    notes = []  # Will contain a sequence of the predicted notes
    sampling_idx = 0
    x = notes_encoded[sampling_idx][0:1, :, :]  # First note of the piece
    notes.append(x.cpu().numpy())  # Saves the first note
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
                if y_pred[:, -1] <= 0.7:  # We add a hold condition, in case the probability
                    choose[:, :, -1] = 1  # of having a hold is close to the one of having the pitch
        if not chosen:
            if print_first:
                print("\nPrinting out the maximum prob of all notes for a time step",
                      "when this maximum prob is less than 0.9")
                print_first = False
            pred_note_idx = np.argmax(y_pred.cpu())
            choose[:, :, pred_note_idx] = 1
            if pred_note_idx != 88:  # No holds for rests
                if y_pred[:, pred_note_idx] - y_pred[:, -1] <= 0.2:  # Hold condition
                    choose[:, :, -1] = 1
            print(_, y_pred[:, np.argmax(y_pred.cpu())])  # Maximum probability out of all components

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

        # TODO: Add condition so it doesn't get stuck....!

        #if _ % (seq_len//2) == 0 and x_new.shape[0] > seq_len//2 + 2:
        #    if sampling_idx >= len(notes_encoded):
        #        sampling_idx = 0
        #    st = randint(1, 100)
        #    for i in range(1, seq_len//2):
        #        x_new[-i] = notes_encoded[sampling_idx][st+i, :, :]
        #    sampling_idx += 1
        #    x = x_new.cuda()

        #if _ % seq_len == 0:
        #    if sampling_idx >= len(notes_encoded):
        #        sampling_idx = 0
        #        change_note = True
        #    if change_note:
        #        x_new[-1] = notes_encoded[sampling_idx][np.random.randint(1, 100, size=1)[0], :, :]
        #        change_note = False
        #    else:
        #        x_new[-1] = notes_encoded[sampling_idx][0, :, :]
        #    sampling_idx += 1
        #    x = x_new.cuda()

    # Gets the notes into the correct NumPy array shape
    gen_notes = np.empty((len(notes)-seq_len+1, 89))  # Doesn't use the first predicted notes
    for idx, nt in enumerate(notes[seq_len-1:]):  # Because at first this will be inaccurate
        gen_notes[idx] = nt[0]  # until an input of seq_len number of sequences is obtained

    # Decodes the generated music and saves it as a MIDI file
    gen_midi = decode(gen_notes, time_step=time_step)
    gen_midi.write("midi", file_name + ".mid")


# notes_encoded = load("mozart", "sonata", 10)

# net, l, ll = train_lstm_loss_whole_seq(50, hidden_size=89, n_epochs=100, lr=0.01)
# torch.save(net.state_dict(), 'lstm_whole_seq_v2_mozart_50.pkl')
# net = LSTMMusic(89, 89).cuda()
# torch.save(net.state_dict(), 'lstm_whole_seq_v2_mozart_50.pkl')
# net.load_state_dict(torch.load("lstm_whole_seq_v2_mozart_50.pkl"))
# net.eval()
# This one doesn't do so well:
# ltsm_gen(net, 50, "mozart_v2", hidden_size=89, time_step=0.25, n_steps=200)  # 
# The following stuff gives decent results: TODO: Fix too many rests
# ltsm_gen_v2(net, 50, "mozart_v2", hidden_size=89, time_step=0.25, n_steps=400)

# net, l, ll = train_lstm_loss_whole_seq(50, hidden_size=160, lr=0.02, n_epochs=100, use_n_seq=30)
# torch.save(net.state_dict(), 'lstm_whole_seq_mozart_50.pkl')
# net = LSTMMusic(89, 160).cuda()
# net.load_state_dict(torch.load("lstm_whole_seq_mozart_50.pkl"))  #
# net.eval()
# ltsm_gen_v2(net, 50, "mozart_v2", hidden_size=160, time_step=0.25, n_steps=400)
# ltsm_gen(net, 50, "mozart_v2", hidden_size=160, time_step=0.25, n_steps=200)

# notes_encoded = load("bach", "unknown", 4)
# net, l, ll = train_lstm_loss_whole_seq(28, hidden_size=160, lr=0.01, n_epochs=100, use_all_seq=True)
# torch.save(net.state_dict(), 'lstm_whole_seq_bach.pkl')
# net = LSTMMusic(89, 160).cuda()
# net.load_state_dict(torch.load("lstm_whole_seq_bach.pkl"))
# net.eval()
# ltsm_gen(net, 50, "bach_v2", hidden_size=160, time_step=0.25, n_steps=200)  # 01
# ltsm_gen_v2(net, 28, "bach_v2", hidden_size=160, time_step=0.25, n_steps=400)  # 02
