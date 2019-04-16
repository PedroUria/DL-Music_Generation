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


def train_lstm(seq_len, hidden_size=89):

    start = time()
    net = LSTMMusic(89, hidden_size).cuda()
    lr = 0.01
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.BCELoss()  # Because we are using sigmoid and not softmax, BCELoss is the right choice
    h_state = torch.zeros(1, 1, hidden_size).float().cuda()  #  Initializes the hidden
    c_state = torch.zeros(1, 1, hidden_size).float().cuda()  #  and cell states
    seq_len = 1  # We will take as input only the previous note
    l = []  # Stores the loss per sequence
    lll = []  # Idem, but will be set to [] after each epoch for ll
    ll = []  #  Stores the mean loss per epoch
    wait_10 = 0  #  We will halve the learning rate if the loss does not decrease in the last 10 epochs
    for epoch in range(100):
        print("---------- epoch number:", epoch, "----------")
        for step in range(notes_encoded.shape[0] // seq_len - 1):
            x = notes_encoded[step:seq_len + step, :, :]  # We take as input the last seq_len notes
            x.requires_grad = True
            y = notes_encoded[step + seq_len:step + seq_len * 2, :, :]  #  Uses the whole next sequence to get the loss
            # y = notes_encoded[step+seq_len, :, :]  # Uses only the next note after input sequence to get the loss
            y_pred, h_c_state = net(x, (h_state, c_state))
            # y_pred = y_pred[0]  # Uses only the next note after input sequence to get the loss
            # y_pred shape: [seq_len, batch_size, hidden_size]
            h_state, c_state = h_c_state[0].data, h_c_state[
                1].data  # repack the hidden state, break the connection from last iteration
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


# net, l, ll = train_lstm(1)  # This takes 499.16 seconds on CPU
#        Last printed outputs:
# ---------- epoch number: 99 ----------
#            loss: 0.0019330173
# Halving learning rate from 0.00015625 to 7.8125e-05
# The training process took 243.24 seconds

# Printing out the maximum prob of all notes for a time step when this maximum prob is less than 0.9
# tensor([0.6974], device='cuda:0')
# tensor([0.3177], device='cuda:0')
# tensor([0.3334], device='cuda:0')
# tensor([0.0479], device='cuda:0')
# tensor([0.7602], device='cuda:0')
# tensor([0.4763], device='cuda:0')
# tensor([0.0533], device='cuda:0')
# tensor([0.0156], device='cuda:0')
# tensor([0.1342], device='cuda:0')
# tensor([0.3906], device='cuda:0')
# tensor([0.4578], device='cuda:0')
# tensor([0.4179], device='cuda:0')
# tensor([0.5913], device='cuda:0')
# tensor([0.8491], device='cuda:0')
# tensor([0.5995], device='cuda:0')
# tensor([0.8802], device='cuda:0')
# tensor([0.6042], device='cuda:0')
# tensor([0.8745], device='cuda:0')
# tensor([0.5978], device='cuda:0')
# tensor([0.8908], device='cuda:0')
# tensor([0.5943], device='cuda:0')
# tensor([0.8994], device='cuda:0')
# tensor([0.5883], device='cuda:0')
# tensor([0.5794], device='cuda:0')
# tensor([0.5647], device='cuda:0')


def plot_loss(l, ll):

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


def ltsm_gen(net, seq_len, file_name, n_steps=100, hidden_size=89):

    notes = []  # Will contain a sequence of the predicted notes
    x = notes_encoded[2:2+seq_len, :, :]  # Input notes
    notes.append(x)
    h_state = torch.zeros(1, 1, hidden_size).float().cuda()
    c_state = torch.zeros(1, 1, hidden_size).float().cuda()
    print_first = True
    for step in range(seq_len, n_steps):
        chosen = False  # To account for when no dimension's probability is bigger than 0.9
        y_pred, h_c_state = net(x, (h_state, c_state))
        h_state, c_state = h_c_state[0].data, h_c_state[1].data
        y_pred = y_pred.data
        choose = torch.zeros(y_pred.shape)  # Coverts the probabilities to the actual notes vectors
        for seq_idx, nt in enumerate(y_pred):
            for idx in range(y_pred.shape[2]):
                if nt[:, idx] > 0.9:
                    choose[seq_idx, :, idx] = 1
                    chosen = True
            if not chosen:
                if print_first:
                    print("\nPrinting out the maximum prob of all notes for a time step",
                          "when this maximum prob is less than 0.9")
                    print_first = False
                choose[:, :, np.argmax(nt)] = 1
                print(nt[:, np.argmax(nt)])  # Maximum probability out of all components
        x = choose.cuda()  # Uses the output of the last time_step as the input for the next time_step
        notes.append(choose)

    # Gets the notes into the correct NumPy array shape
    gen_notes = np.empty((len(notes), 89))
    for idx, nt in enumerate(notes):
        gen_notes[idx] = nt.cpu().numpy()[0]

    # Decodes the generated music and saves it as a MIDI file
    gen_midi = decode(gen_notes, time_step=0.25)
    gen_midi.write("midi", file_name + ".mid")


# ltsm_gen(net, 1, "come_on")


# ------------------------------------------
# Let's do the same with a seq_len = 14
# ------------------------------------------

# print("\nNow let's use sequences of 14 time steps as inputs",
#       "to train the network and also to generate the music\n")
# net, l, ll = train_lstm(14)
#        Last printed outputs:
# Halving learning rate from 0.00015625 to 7.8125e-05
# ---------- epoch number: 98 ----------
#            loss: 0.0023078104
# ---------- epoch number: 99 ----------
#            loss: 0.002350083
# The training process took 244.88 seconds
# Printing out the maximum prob of all notes for a time step when this maximum prob is less than 0.9
# tensor([0.6940], device='cuda:0')
# tensor([0.1850], device='cuda:0')
# tensor([0.0057], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([6.6648e-07], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([3.0197e-07], device='cuda:0')
# tensor([6.2799e-07], device='cuda:0')
# tensor([7.9442e-09], device='cuda:0')
# tensor([1.1805e-07], device='cuda:0')
# tensor([3.8359e-09], device='cuda:0')
# tensor([1.6552e-10], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.0054], device='cuda:0')
# tensor([0.4542], device='cuda:0')
# tensor([0.2610], device='cuda:0')
# tensor([0.2126], device='cuda:0')
# tensor([0.4800], device='cuda:0')
# tensor([0.6105], device='cuda:0')
# tensor([0.0002], device='cuda:0')
# tensor([0.5819], device='cuda:0')
# tensor([0.6570], device='cuda:0')
# tensor([0.0137], device='cuda:0')
# tensor([0.6101], device='cuda:0')
# tensor([0.0412], device='cuda:0')
# tensor([0.0290], device='cuda:0')
# tensor([0.1456], device='cuda:0')
# tensor([0.0088], device='cuda:0')
# tensor([0.2581], device='cuda:0')
# tensor([0.2962], device='cuda:0')
# tensor([0.0022], device='cuda:0')
# tensor([2.7644e-07], device='cuda:0')
# tensor([0.2341], device='cuda:0')
# tensor([0.0071], device='cuda:0')
# tensor([0.0462], device='cuda:0')
# tensor([0.1765], device='cuda:0')
# tensor([0.1366], device='cuda:0')
# tensor([0.0053], device='cuda:0')
# tensor([0.1197], device='cuda:0')
# tensor([0.0003], device='cuda:0')
# tensor([0.0005], device='cuda:0')
# tensor([0.1256], device='cuda:0')
# tensor([0.0250], device='cuda:0')
# tensor([0.0553], device='cuda:0')
# tensor([0.1841], device='cuda:0')
# tensor([0.1002], device='cuda:0')
# tensor([0.0122], device='cuda:0')
# tensor([0.7283], device='cuda:0')
# tensor([0.1948], device='cuda:0')
# tensor([0.0124], device='cuda:0')
# tensor([0.0089], device='cuda:0')
# tensor([0.0397], device='cuda:0')
# tensor([0.0023], device='cuda:0')
# tensor([0.0015], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.0196], device='cuda:0')
# tensor([0.0007], device='cuda:0')
# tensor([0.0110], device='cuda:0')
# tensor([0.1301], device='cuda:0')
# tensor([0.7965], device='cuda:0')
# tensor([0.3816], device='cuda:0')
# tensor([0.0017], device='cuda:0')
# tensor([0.3082], device='cuda:0')
# tensor([0.1198], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.0009], device='cuda:0')
# tensor([0.1267], device='cuda:0')
# tensor([0.0325], device='cuda:0')
# tensor([0.2682], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.0021], device='cuda:0')
# tensor([0.6413], device='cuda:0')
# tensor([0.0004], device='cuda:0')
# tensor([0.0116], device='cuda:0')
# tensor([0.0009], device='cuda:0')
# tensor([0.0431], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.0016], device='cuda:0')
# tensor([0.0080], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([1.4275e-06], device='cuda:0')
# tensor([0.0007], device='cuda:0')
# tensor([4.7314e-10], device='cuda:0')
# tensor([4.5893e-06], device='cuda:0')
# tensor([1.1894e-09], device='cuda:0')
# tensor([0.0003], device='cuda:0')
# tensor([8.7942e-06], device='cuda:0')
# tensor([1.3923e-10], device='cuda:0')
# tensor([1.2478e-06], device='cuda:0')
# tensor([8.6986e-06], device='cuda:0')
# tensor([3.0259e-09], device='cuda:0')
# tensor([1.1546e-07], device='cuda:0')
# tensor([2.4846e-09], device='cuda:0')
# tensor([0.5836], device='cuda:0')
# tensor([0.1446], device='cuda:0')
# tensor([0.0991], device='cuda:0')
# tensor([0.2129], device='cuda:0')
# tensor([0.0050], device='cuda:0')
# tensor([0.1479], device='cuda:0')
# tensor([0.3322], device='cuda:0')
# tensor([0.3432], device='cuda:0')
# tensor([0.0049], device='cuda:0')
# tensor([0.0439], device='cuda:0')
# tensor([0.0011], device='cuda:0')
# tensor([0.0003], device='cuda:0')
# tensor([0.0566], device='cuda:0')
# tensor([0.0509], device='cuda:0')
# tensor([0.0129], device='cuda:0')
# tensor([0.0145], device='cuda:0')
# tensor([0.2035], device='cuda:0')
# tensor([0.0132], device='cuda:0')
# tensor([0.4008], device='cuda:0')
# tensor([0.0702], device='cuda:0')
# tensor([0.0007], device='cuda:0')
# tensor([0.0116], device='cuda:0')
# tensor([0.0061], device='cuda:0')
# tensor([0.3114], device='cuda:0')
# tensor([0.4689], device='cuda:0')
# tensor([0.7530], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([8.0868e-06], device='cuda:0')
# tensor([3.2016e-07], device='cuda:0')
# tensor([1.3529e-07], device='cuda:0')
# tensor([3.5820e-07], device='cuda:0')
# tensor([5.2864e-08], device='cuda:0')
# tensor([4.1845e-08], device='cuda:0')
# tensor([1.8608e-08], device='cuda:0')
# tensor([2.2753e-09], device='cuda:0')
# tensor([8.8127e-09], device='cuda:0')
# tensor([1.6006e-11], device='cuda:0')
# tensor([2.2887e-08], device='cuda:0')
# tensor([3.7999e-08], device='cuda:0')
# tensor([7.9729e-09], device='cuda:0')
# tensor([0.7941], device='cuda:0')
# tensor([0.0023], device='cuda:0')
# tensor([0.2133], device='cuda:0')
# tensor([0.2993], device='cuda:0')
# tensor([0.0138], device='cuda:0')
# tensor([0.0701], device='cuda:0')
# tensor([0.6260], device='cuda:0')
# tensor([0.0080], device='cuda:0')
# tensor([0.0190], device='cuda:0')
# tensor([0.1706], device='cuda:0')
# tensor([0.0823], device='cuda:0')
# tensor([0.1496], device='cuda:0')
# tensor([0.7488], device='cuda:0')
# tensor([0.0620], device='cuda:0')
# tensor([0.8078], device='cuda:0')
# tensor([0.6518], device='cuda:0')
# tensor([0.0163], device='cuda:0')
# tensor([0.0004], device='cuda:0')
# tensor([0.0145], device='cuda:0')
# tensor([0.2867], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.0020], device='cuda:0')
# tensor([0.0069], device='cuda:0')
# tensor([1.4507e-07], device='cuda:0')
# tensor([0.0169], device='cuda:0')
# tensor([0.0002], device='cuda:0')
# tensor([7.1132e-06], device='cuda:0')
# tensor([0.0004], device='cuda:0')
# tensor([1.6159e-07], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.0079], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.0246], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([2.7607e-06], device='cuda:0')
# tensor([5.4018e-06], device='cuda:0')
# tensor([0.0021], device='cuda:0')
# tensor([0.0012], device='cuda:0')
# tensor([0.0002], device='cuda:0')
# tensor([0.0012], device='cuda:0')
# tensor([3.9852e-07], device='cuda:0')
# tensor([7.0178e-06], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.0003], device='cuda:0')
# tensor([2.0579e-07], device='cuda:0')
# tensor([0.0289], device='cuda:0')
# tensor([0.5468], device='cuda:0')
# tensor([0.5508], device='cuda:0')
# tensor([0.2091], device='cuda:0')
# tensor([0.2946], device='cuda:0')
# tensor([0.1751], device='cuda:0')
# tensor([0.7502], device='cuda:0')
# tensor([0.1667], device='cuda:0')
# tensor([0.4321], device='cuda:0')
# tensor([0.1428], device='cuda:0')
# tensor([0.7015], device='cuda:0')
# tensor([0.6140], device='cuda:0')
# tensor([0.4763], device='cuda:0')
# tensor([0.2970], device='cuda:0')
# tensor([0.3379], device='cuda:0')
# tensor([0.1204], device='cuda:0')
# tensor([0.8122], device='cuda:0')
# tensor([0.0003], device='cuda:0')
# tensor([0.0209], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.1533], device='cuda:0')
# tensor([0.0342], device='cuda:0')
# tensor([0.7660], device='cuda:0')
# tensor([0.5075], device='cuda:0')
# tensor([0.1021], device='cuda:0')
# tensor([0.2814], device='cuda:0')
# tensor([0.0017], device='cuda:0')
# tensor([0.3695], device='cuda:0')
# tensor([0.8571], device='cuda:0')
# tensor([0.0207], device='cuda:0')
# tensor([0.3489], device='cuda:0')
# tensor([0.0465], device='cuda:0')
# tensor([0.0613], device='cuda:0')
# tensor([0.1354], device='cuda:0')
# tensor([0.0557], device='cuda:0')
# tensor([0.7379], device='cuda:0')
# tensor([0.6161], device='cuda:0')
# tensor([0.0626], device='cuda:0')
# tensor([3.6099e-06], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.8569], device='cuda:0')
# tensor([0.4080], device='cuda:0')
# tensor([0.7021], device='cuda:0')
# tensor([0.7879], device='cuda:0')
# tensor([0.1439], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.0002], device='cuda:0')
# tensor([0.2139], device='cuda:0')
# tensor([0.0498], device='cuda:0')
# tensor([0.2479], device='cuda:0')
# tensor([0.0022], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.0192], device='cuda:0')
# tensor([0.0050], device='cuda:0')
# tensor([0.0062], device='cuda:0')
# tensor([0.0541], device='cuda:0')
# tensor([8.8916e-09], device='cuda:0')
# tensor([0.8110], device='cuda:0')
# tensor([0.4588], device='cuda:0')
# tensor([0.0329], device='cuda:0')
# tensor([0.6621], device='cuda:0')
# tensor([0.0086], device='cuda:0')
# tensor([0.8273], device='cuda:0')
# tensor([0.3275], device='cuda:0')
# tensor([0.8795], device='cuda:0')
# tensor([0.0018], device='cuda:0')
# tensor([0.1971], device='cuda:0')
# tensor([0.0302], device='cuda:0')
# tensor([0.6787], device='cuda:0')
# tensor([0.4555], device='cuda:0')
# tensor([0.4695], device='cuda:0')
# tensor([0.2966], device='cuda:0')
# tensor([0.0116], device='cuda:0')
# tensor([0.3133], device='cuda:0')
# tensor([0.0023], device='cuda:0')
# tensor([1.5310e-08], device='cuda:0')
# tensor([0.0011], device='cuda:0')
# tensor([1.0299e-07], device='cuda:0')
# tensor([2.6291e-06], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([4.1578e-09], device='cuda:0')
# tensor([4.6721e-08], device='cuda:0')
# tensor([1.8840e-09], device='cuda:0')
# tensor([0.7930], device='cuda:0')
# tensor([0.3218], device='cuda:0')
# tensor([0.5647], device='cuda:0')
# tensor([0.0181], device='cuda:0')
# tensor([0.0374], device='cuda:0')
# tensor([0.1205], device='cuda:0')
# tensor([0.3280], device='cuda:0')
# tensor([0.0488], device='cuda:0')
# tensor([0.6983], device='cuda:0')
# tensor([0.4135], device='cuda:0')
# tensor([0.6117], device='cuda:0')
# tensor([0.0110], device='cuda:0')
# tensor([0.0745], device='cuda:0')
# tensor([0.0553], device='cuda:0')
# tensor([0.3972], device='cuda:0')
# tensor([0.8619], device='cuda:0')
# tensor([0.1876], device='cuda:0')
# tensor([0.1979], device='cuda:0')
# tensor([0.2005], device='cuda:0')
# tensor([0.8422], device='cuda:0')
# tensor([0.7416], device='cuda:0')
# tensor([0.2168], device='cuda:0')
# tensor([0.8077], device='cuda:0')
# tensor([0.7751], device='cuda:0')
# tensor([0.7019], device='cuda:0')
# tensor([0.8523], device='cuda:0')
# tensor([0.7844], device='cuda:0')
# tensor([0.8156], device='cuda:0')
# tensor([0.0520], device='cuda:0')
# tensor([0.2079], device='cuda:0')
# tensor([0.0493], device='cuda:0')
# tensor([0.6528], device='cuda:0')
# tensor([0.8042], device='cuda:0')
# tensor([0.5844], device='cuda:0')
# tensor([0.0095], device='cuda:0')
# tensor([0.4586], device='cuda:0')
# tensor([0.0344], device='cuda:0')
# tensor([0.0153], device='cuda:0')
# tensor([0.0490], device='cuda:0')
# tensor([0.2599], device='cuda:0')
# tensor([0.1348], device='cuda:0')
# tensor([0.1452], device='cuda:0')
# tensor([0.6523], device='cuda:0')
# tensor([0.4425], device='cuda:0')
# tensor([0.0096], device='cuda:0')
# tensor([0.1616], device='cuda:0')
# tensor([0.0260], device='cuda:0')
# tensor([0.0310], device='cuda:0')
# tensor([0.0004], device='cuda:0')
# tensor([0.0010], device='cuda:0')
# tensor([0.0053], device='cuda:0')
# This is certainly a problem
# TODO: Figure out if there is something wrong when imputing sequences
# TODO: Or if it has to do with the way it is using the predicted probs
# plot_loss(l, ll)
# ltsm_gen(net, 14, "come_on_14")

# ------------------------------------------
# Let's do the same with a seq_len = 14
# and double the number of neurons on the
# LSTM Layer
# ------------------------------------------
# print("\nDoing the same as before, but with 178 hidden neurons instead of 89\n")
# net, l, ll = train_lstm(14, hidden_size=178)
#        Last printed outputs:
# ---------- epoch number: 99 ----------
#            loss: 0.0018877849
# Halving learning rate from 7.8125e-05 to 3.90625e-05
# The training process took 249.3 seconds
#
# Printing out the maximum prob of all notes for a time step when this maximum prob is less than 0.9
# tensor([0.8514], device='cuda:0')
# tensor([0.1061], device='cuda:0')
# tensor([0.0033], device='cuda:0')
# tensor([0.0007], device='cuda:0')
# tensor([0.1151], device='cuda:0')
# tensor([0.2607], device='cuda:0')
# tensor([0.0274], device='cuda:0')
# tensor([0.2605], device='cuda:0')
# tensor([0.1083], device='cuda:0')
# tensor([0.6987], device='cuda:0')
# tensor([0.0448], device='cuda:0')
# tensor([0.0115], device='cuda:0')
# tensor([0.0067], device='cuda:0')
# tensor([0.0333], device='cuda:0')
# tensor([0.0107], device='cuda:0')
# tensor([0.6686], device='cuda:0')
# tensor([0.0608], device='cuda:0')
# tensor([0.8795], device='cuda:0')
# tensor([0.0311], device='cuda:0')
# tensor([0.2888], device='cuda:0')
# tensor([0.0273], device='cuda:0')
# tensor([0.0008], device='cuda:0')
# tensor([0.0014], device='cuda:0')
# tensor([0.4029], device='cuda:0')
# tensor([0.5932], device='cuda:0')
# tensor([0.7539], device='cuda:0')
# tensor([0.4538], device='cuda:0')
# tensor([0.0067], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.4013], device='cuda:0')
# tensor([0.4599], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.3526], device='cuda:0')
# tensor([0.0598], device='cuda:0')
# tensor([0.1617], device='cuda:0')
# tensor([0.0085], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.0006], device='cuda:0')
# tensor([0.0583], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.0176], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.0017], device='cuda:0')
# tensor([0.0002], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.6673], device='cuda:0')
# tensor([0.0415], device='cuda:0')
# tensor([0.1241], device='cuda:0')
# tensor([0.8953], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.4109], device='cuda:0')
# tensor([0.2240], device='cuda:0')
# tensor([0.0217], device='cuda:0')
# tensor([0.0004], device='cuda:0')
# tensor([0.2052], device='cuda:0')
# tensor([0.0007], device='cuda:0')
# tensor([0.0004], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([1.2334e-09], device='cuda:0')
# tensor([2.3556e-09], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.8224], device='cuda:0')
# tensor([1.6316e-14], device='cuda:0')
# tensor([0.0002], device='cuda:0')
# tensor([1.0784e-10], device='cuda:0')
# tensor([6.9122e-07], device='cuda:0')
# tensor([0.3531], device='cuda:0')
# tensor([0.1069], device='cuda:0')
# tensor([0.5255], device='cuda:0')
# tensor([0.7438], device='cuda:0')
# tensor([0.4344], device='cuda:0')
# tensor([0.0231], device='cuda:0')
# tensor([0.0520], device='cuda:0')
# tensor([1.6127e-06], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.7989], device='cuda:0')
# tensor([0.0563], device='cuda:0')
# tensor([0.0021], device='cuda:0')
# tensor([0.4710], device='cuda:0')
# tensor([0.1089], device='cuda:0')
# tensor([0.0078], device='cuda:0')
# tensor([0.0025], device='cuda:0')
# tensor([0.0072], device='cuda:0')
# tensor([0.0400], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.0011], device='cuda:0')
# tensor([0.0011], device='cuda:0')
# tensor([0.1305], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.0012], device='cuda:0')
# tensor([0.0184], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.0002], device='cuda:0')
# tensor([0.0002], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.6517], device='cuda:0')
# tensor([0.0188], device='cuda:0')
# tensor([0.0062], device='cuda:0')
# tensor([0.8999], device='cuda:0')
# tensor([0.4428], device='cuda:0')
# tensor([0.0946], device='cuda:0')
# tensor([2.4113e-06], device='cuda:0')
# tensor([0.0015], device='cuda:0')
# tensor([9.3749e-11], device='cuda:0')
# tensor([1.7194e-14], device='cuda:0')
# tensor([1.1085e-07], device='cuda:0')
# tensor([8.7496e-09], device='cuda:0')
# tensor([7.2633e-12], device='cuda:0')
# tensor([1.6135e-12], device='cuda:0')
# tensor([1.3033e-10], device='cuda:0')
# tensor([1.3486e-14], device='cuda:0')
# tensor([1.0583e-12], device='cuda:0')
# tensor([0.5828], device='cuda:0')
# tensor([0.0237], device='cuda:0')
# tensor([0.2254], device='cuda:0')
# tensor([0.0014], device='cuda:0')
# tensor([1.1633e-07], device='cuda:0')
# tensor([0.2968], device='cuda:0')
# tensor([0.8303], device='cuda:0')
# tensor([0.3002], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.0081], device='cuda:0')
# tensor([0.0367], device='cuda:0')
# tensor([0.0198], device='cuda:0')
# tensor([0.3313], device='cuda:0')
# tensor([0.3588], device='cuda:0')
# tensor([0.1037], device='cuda:0')
# tensor([0.4220], device='cuda:0')
# tensor([0.0877], device='cuda:0')
# tensor([0.0005], device='cuda:0')
# tensor([0.0266], device='cuda:0')
# tensor([0.1495], device='cuda:0')
# tensor([0.0121], device='cuda:0')
# tensor([0.2955], device='cuda:0')
# tensor([0.0048], device='cuda:0')
# tensor([0.3663], device='cuda:0')
# tensor([0.3442], device='cuda:0')
# tensor([4.0042e-06], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.0759], device='cuda:0')
# tensor([0.0691], device='cuda:0')
# tensor([0.2577], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.0012], device='cuda:0')
# tensor([0.0436], device='cuda:0')
# tensor([0.0225], device='cuda:0')
# tensor([0.8281], device='cuda:0')
# tensor([0.4504], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([0.7965], device='cuda:0')
# tensor([0.0340], device='cuda:0')
# tensor([0.0280], device='cuda:0')
# tensor([0.7467], device='cuda:0')
# tensor([0.4049], device='cuda:0')
# tensor([0.0004], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([3.5711e-08], device='cuda:0')
# tensor([2.1412e-11], device='cuda:0')
# tensor([6.1919e-07], device='cuda:0')
# tensor([0.0007], device='cuda:0')
# tensor([0.1737], device='cuda:0')
# tensor([0.0980], device='cuda:0')
# tensor([0.7506], device='cuda:0')
# tensor([0.1536], device='cuda:0')
# tensor([0.0167], device='cuda:0')
# tensor([2.0277e-10], device='cuda:0')
# tensor([8.6061e-07], device='cuda:0')
# tensor([0.0001], device='cuda:0')
# tensor([2.1797e-09], device='cuda:0')
# tensor([0.0039], device='cuda:0')
# tensor([0.0095], device='cuda:0')
# tensor([0.0026], device='cuda:0')
# tensor([7.0686e-11], device='cuda:0')
# tensor([5.2387e-07], device='cuda:0')
# tensor([0.0052], device='cuda:0')
# tensor([0.5322], device='cuda:0')
# tensor([0.0004], device='cuda:0')
# tensor([0.6701], device='cuda:0')
# tensor([0.6597], device='cuda:0')
# tensor([0.8364], device='cuda:0')
# plot_loss(l, ll)
# ltsm_gen(net, 14, "come_on_14_double_neur", hidden_size=178)


# ------------------------------------------
# Let's do the same with a seq_len = 1
# and double the number of neurons on the
# LSTM Layer
# ------------------------------------------
print("\nDoing seq_len=1 and with 178 hidden neurons instead of 89\n")
net, l, ll = train_lstm(1, hidden_size=178)
#        Last printed outputs:
# ---------- epoch number: 99 ----------
#            loss: 0.0016991186
# Halving learning rate from 0.00015625 to 7.8125e-05
# The training process took 258.29 seconds
#
# Printing out the maximum prob of all notes for a time step when this maximum prob is less than 0.9
# tensor([0.7229], device='cuda:0')
# tensor([0.5365], device='cuda:0')
# tensor([0.7964], device='cuda:0')
# tensor([0.1156], device='cuda:0')
# tensor([2.7928e-06], device='cuda:0')
# tensor([9.4554e-11], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.0000], device='cuda:0')
# tensor([0.5601], device='cuda:0')
# tensor([0.6314], device='cuda:0')
# tensor([0.8101], device='cuda:0')
# tensor([0.7183], device='cuda:0')
# tensor([0.4454], device='cuda:0')
# tensor([0.8080], device='cuda:0')
# tensor([0.5760], device='cuda:0')
# tensor([0.6486], device='cuda:0')
# tensor([0.6023], device='cuda:0')
# tensor([0.2090], device='cuda:0')
# tensor([0.2344], device='cuda:0')
# tensor([0.8578], device='cuda:0')
# tensor([0.8671], device='cuda:0')
# tensor([0.6543], device='cuda:0')
# tensor([0.8610], device='cuda:0')
# tensor([0.7026], device='cuda:0')
# tensor([0.2999], device='cuda:0')
plot_loss(l, ll)
ltsm_gen(net, 1, "come_on_double_neur", hidden_size=178)


# TODO: Before using many files, we should figure out if there is a way to get
# reproducible results (random seed), but I do not think there is, given the
# nature of LSTMs. Also, need to figure out doing mini-batching and last but not
# least, solve the limitation of generating a sequence of the same length as the
# input sequence, and also use only one note as input but then use sequences on previous
# notes on the following iterations... Need to figure out sequence length and its relationship
# with delays...
# Last but not least, figure out the problem with the sequence stuff, and maybe get a better optimizer
