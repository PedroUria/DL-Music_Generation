# Music-Generation

This repo consists on the final proyect by [PedroUria](https://github.com/PedroUria), [QuirkyDataScientist1978](https://github.com/QuirkyDataScientist1978) and [thekartikay](https://github.com/thekartikay) for our Neural Network class. Below is a description of its structure and instructions to use the code to train LSTM networks and/or generate music using our trained networks. You can also visit [www.lstmmusic.com](http://www.lstmmusic.com/) to listen to some samples and generate music by interacting with a simple UI. However, if you want to train your own models, you should stay here. We also have a [playlist](https://soundcloud.com/pedro-uria-193566069/sets/classical-846-lstm).

## Structure

The repo is organized in various subdirectories. [code](code) contains all the code used in the project, while [data](data) contains all the data used to trained the networks. [Group-Proposal](Group-Proposal) contains the project proposal, while [Final-Group-Presentation](Final-Group-Presentation) contains the presentation slides we used in class and [Final-Group-Project-Report](Final-Group-Project-Report) contains the report, which is the best source to understanding the project, although [Final-Group-Presentation](Final-Group-Presentation) may be enough. Some of our results are located in [generated-music](generated-music). 

## Instructions

### Dependencies

#### Software

- [Python 3.5.2](https://www.python.org/downloads/release/python-352/): all the code is written in Python.

- [MuseScore](https://musescore.org/en): to open `.mid` files and show music scores using `music21`.

#### Python Packages

##### Basic

- [`os`](https://docs.python.org/2/library/os.html): to navigate the different folders.
- [`time`](https://docs.python.org/2/library/time.html): to time some processes.
- [`random`](https://docs.python.org/2/library/random.html): for an optional feature on the generator functions.

##### External

- [`music21`](https://web.mit.edu/music21/doc/), version 5.5.0: To read and write `.mid` files.
- [`NumPy`](https://docs.scipy.org/doc/), version 1.15.0: To encode the `.mid` files.
- [`PyTorch`](https://pytorch.org/docs/stable/index.html), version 0.4.1: to build and train the networks.
- [`matplotlyb`](https://matplotlib.org/contents.html), version 1.5.1: to plot losses.

#### Some Notes

All the code in this project, apart from a Jupyter Notebook that served as the starting ground for getting familiar with `music21`, was run on Google Cloud Platform on an ubuntu instance via a custom image provided by our professor. To install `music21` on this instance, and possibly on any ubuntu vm, you need to run `python3 -m pip install --user music21` on the terminal. The ubuntu version was 16.04, code name xenial. However, by creating a virtual enviroment with the software, packages and versions listed above, there should not be any issues. The code will also run the networks on GPU automatically if you have access to one.

### Training your own networks

If you want to experiment training your own networks on our data or any other data, you can use the scripts on [code](code) that are named as `training_<something>.py`. We found [training\_many\_deff\_two\_voices\_stacked.py](code/training_many_deff_two_voices_stacked.py) to be the most successful in general. There are many hyperparameters you can play with inside these scripts, regarding the network arquitecture, the training process and the generation process. The scripts are written to take in data from [data/classical](data/classical), but some easy tweaks would allow to take in any other `.mid` files. You can read [code/README.me](code/README.me) and also the functions documentations, and refer to [Final-Group-Project-Report/Music-LSTM.pdf](Final-Group-Project-Report/Music-LSTM.pdf) and [Final-Group-Presentation/slides.pdf](Final-Group-Presentation/slides.pdf) to know what is going on.

### Using our trained networks to generate new music

You can also use some of the models we saved in [generated-music](generated-music) by following the instructions in this folder. The are two generator functions, with many hyperparameters, so you can obtain a lot of variations even when playing with the same model.











