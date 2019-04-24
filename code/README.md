This subdirectory contains all the code used in this project. There are three main parts to it, although two of them are written on the same scripts.

## Encoding and Decoding `.mid` files

### Encoder & Decoder

[Data_Preprocessing.ipynb](Data_Preprocessing.ipynb) consitst on a brief overview of the `music21` package as well as the process of encoding and decoding the `.mid` files. The actual encoder and decoder functions are located on [encoder_decoder.py](encoder_decoder.py). `encode()` takes as input one hand of a musical piece, either right or left, and conversts it into a matrix, which is actually a sequence of rows where each row is a multy-hot-encoded vector of each time event (note). The time step is set as a 64th note by default, although we mostly used a time step of a 16th note when training the networks. `decode()` does the opposite: takes as input a matrix of encoded notes and converts it back to a `music21.stream.Stream()` object, which then can be read saved as a `.mid` file. 

### Both hands together

Both of these work on only one hand/voice. In order to encode two hands together (stacking them horizontally and creating a sequence of vectors with 2x components), we use a function named `get_both_hands`, which is defined inside the scripts dedicated to generate music with both harmony and melody (the ones containing `two_voices` on their names. To decode two hands stacked together, we use `decode()` on each hand separately and then use `combine()` located in [combine.py](combine.py) to get back the `music21.stream.Stream()` object.

We also wrote another approach, [multi_many_hot_encoder_decoder.py](multi_many_hot_encoder_decoder.py), which was finally not used (you can read more about it on the report).

### Evaluating quality

[eval_processor.py](eval_processor.py) was an attempt to evaluate the quality of the `encode()` and `decode()` functions, although it was abandoned because it did not make much sense and there were too many things to consider, and thus deemed not worth the trouble. Instead, we evaluated the functions trying out different songs manually and comparing the output of `decode(encode())` with the original files. The functions do have some bugs that were found during training, but for most of the files they work perfectly.

## Building & Training the Networks

### On melodies

The code for building training the networks was written in phases. First, we developed the arquitecture and the training process to train on only the melody and only one song, i.e, the most simplest case. We wanted to see how well our network could learn a given song (even if this meant overfiting on it). The code for this is located on [training\_many\_deff.py](training_many_deff.py), which actually can also take more than one song as training data. We used this script to train a LSTM network on [bach_846.mid](data/classical/bach/unknown/bach_846.mid), and generate a very similar output ([bach_4.mid](bach_4.mid)) using the trained network and the first note of [bach_846.mid](data/classical/bach/unknown/bach_846.mid) as the only input. The generative functions are also located inside this script, but will be described later. Then, we also used this script to train on many different songs.

### On melodies + harmonies

After this, we built our networks to be trained and generate both the melody and the harmony at the same time. We used three different approaches, documented in the report and named like `training\_many\_something`.

## Generating Music

To generate music, you can just run the example code on the training files. There are also instructions in the generated music folder.
