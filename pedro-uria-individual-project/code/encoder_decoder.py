import music21 as ms
import numpy as np


def encode(hand, time_step=0.05):
    
    """
    Many-hot-encoding of a hand of a musical piece
    :param hand: a music21 stream for piano instrument
    :returns: encoded ordered temporal vectors
    Each vector corresponds to a time_step=64th
    Components 0 to 87 are the notes in ascending
    frequency order. Component 88 = 1 if there is
    a rest. Component 89 indicates hold, to account
    for notes with duration longer than 64th. Last
    component (90) is the tempo, with numerical values
    indicating the speed at which the music piece is being played
    Even if the tempo changes at the middle of the piece,
    this function sets the same tempo for all notes in the piece.
    NOTE: This works on one hand only (stream which 
    does not contain any other streams)
    """
    
    # Gets a NumPy array with all the frequency of the piano notes
    notes_freq = [ms.note.Note(note).pitch.frequency for note in ["A0", "A#0"]]
    s = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i in range(1, 8):
        notes_freq += [ms.note.Note(note + str(i)).pitch.frequency for note in s]
    notes_freq.append(ms.note.Note("C8").pitch.frequency)
    notes_freq = np.array(notes_freq)
    
    # Gets the time step. Each vector will contain note/s of 16th length 
    time_step = time_step
    
    # Gets the numpy array to store all the encoded notes, rests, holds and tempo
    for nt in hand[::-1]:  # To do so, we need to get the offset
        try:  # of the last note/rest, together with its duration
            size = nt.offset + nt.duration.quarterLength
            break
        except:
            pass
    notes = np.zeros((int(size/time_step), 90))
    
    # Encodes all the music
    # This flag will be used to only get the tempo of the beginning of the piece
    temp_flag = True  # And assign this tempo to all the notes in the piece
    idx = -1  # To only append notes, rests or chords
    for nt in hand:
        if type(nt) == ms.note.Note or type(nt) == ms.note.Rest or type(nt) == ms.chord.Chord:
            idx = int(nt.offset/time_step)  # Temporal index
        # Tempo Encoding
        if temp_flag:
            if type(nt) == ms.tempo.MetronomeMark:
                notes[:, 89] = nt.number
                temp_flag = False
        if idx >= 0:
            # Loops over the duration of the note (if a note is 32nd
            # we need to put this note in two consecutive vector)
            n_vectors = int(nt.duration.quarterLength/time_step)  # 0.33 will be 6 vectors
            # when the actual number would be 0.33/0.05 = 6.6 vectors
            for i in range(n_vectors):
                # Note encoding: one/many-hot encoding
                if type(nt) == ms.note.Note:
                    notes[idx + i, :87] += (notes_freq == nt.pitch.frequency)*1
                # Rest Encoding: one-hot encoding
                if type(nt) == ms.note.Rest:
                    notes[idx + i, 87] = 1
                # Chord Encoding: many-hot encoding
                if type(nt) == ms.chord.Chord:
                    for freqs in [nts.frequency for nts in nt.pitches]:
                        notes[idx + i, :87] += (notes_freq == freqs)*1
                # Hold Encoding: If the duration of the note is longer than time_step
                notes[idx + i, 88] = 1
            try:
                notes[idx + i, 88] = 0
            except:
                notes[idx + i - 2, 88] = 0
        idx = -1
            
    return notes


def decode(notes_encoded, tempo=74, time_step=0.05):
    
    """
    Returns the encoded notes by encode()
    to the original music21 notation
    to get the MIDI file back
    :param notes_encoded: ordered temporal
    vectors in a 2D NumPy array, where
    each row representes the music at certain time_step
    :param: tempo
    :returns: music21 stream object
    """
    
    # Gets the music21 letter representation of the notes pitches
    notes_letters = ["A0", "A#0"]
    s = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i in range(1, 8):
        notes_letters += [note + str(i) for note in s]
    notes_letters.append("C8")
    # Gets a NumPy array with all the frequency of the piano notes
    notes_freq = [ms.note.Note(note).pitch.frequency for note in ["A0", "A#0"]]
    s = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i in range(1, 8):
        notes_freq += [ms.note.Note(note + str(i)).pitch.frequency for note in s]
    notes_freq.append(ms.note.Note("C8").pitch.frequency)
    notes_freq = np.array(notes_freq)
    
    # Creates the stream object and appends some default stuff 
    stream = ms.stream.Stream()
    stream.append(ms.instrument.Piano())
    stream.append(ms.tempo.MetronomeMark(number=tempo))
    stream.append(ms.key.Key("C"))
    stream.append(ms.meter.TimeSignature())

    if notes_encoded.shape[1] == 90:
        hold_index = -2
    else:
        hold_index = -1
    
    time_step = time_step
    offset = 0
    hold = {}  # Will store the duration of the hold for each note
    for j, p in enumerate(notes_encoded):
        if p[87]:  # If we have a Rest, do the same as below
            try:
                if offset < hold[87]:
                    offset += time_step
                    continue
            except:
                pass
            nt = ms.note.Rest()
            dur = time_step
            if p[hold_index]:  
                i = j + 1
                if i < len(notes_encoded) - 1:
                    dur += time_step
                    while notes_encoded[i][hold_index]:  
                        dur += time_step
                        if i < len(notes_encoded) - 1: 
                            i += 1
                        else:
                            break
            hold[87] = dur + offset  
            nt.duration = ms.duration.Duration(dur)
            stream.append(nt)
            stream[-1].offset = offset
        else:
            # For each note on the vector at time = offset
            for frequ_index in np.nonzero(notes_freq * p[:87])[0]:
                try:  # If the duration of the hold + offset is longer than the current offset
                    if offset < hold[frequ_index]:
                        continue  # Do not append this note to the stream
                except:  # As it will be the same note appended on the first previous iteration
                    pass
                # Gets the pitch for a note in p
                nt = ms.note.Note(notes_letters[int(frequ_index)])
                # Gets the duration
                dur = time_step
                if p[hold_index]:  # If we have a hold
                    i = j + 1  # Move onto the next p vector
                    if i < len(notes_encoded) - 1:
                        dur += time_step
                        while notes_encoded[i][-2]:  # And do so until the hold disappears
                            dur += time_step
                            if i < len(notes_encoded) - 1:  # In case there is a hold on the last note
                                i += 1
                            else:
                                break
                hold[frequ_index] = dur + offset  # Total duration of the hold, from the offset
                nt.duration = ms.duration.Duration(dur)
                # Appends to the stream
                stream.append(nt)
                # Sets the offset (need to do it here)
                stream[-1].offset = offset

        offset += time_step
                
    # Accounts for chords
    stream_with_chords = ms.stream.Stream()
    stream_with_chords.append(stream[0])
    le = len(stream)
    for i in range(len(stream)-1):
        if type(stream[i]) == ms.note.Note:
            if (stream[i+1].duration.quarterLength, stream[i+1].offset) == (stream[i].duration.quarterLength, stream[i].offset):
                j = i + 1
                notes_in_chord = [stream[i].nameWithOctave]
                while (stream[j].duration.quarterLength, stream[j].offset) == (stream[i].duration.quarterLength, stream[i].offset):
                    notes_in_chord.append(stream[j].nameWithOctave)
                    j += 1
                    if j == le:
                        break
                stream_with_chords.append(ms.chord.Chord(notes_in_chord))
                stream_with_chords[-1].duration = ms.duration.Duration(stream[i].duration.quarterLength)
                stream_with_chords[-1].offset = stream[i].offset
                stream_with_chords.pop(-2)
            else:
                stream_with_chords.append(stream[i+1])
                stream_with_chords[-1].offset = stream[i+1].offset
        else:
            stream_with_chords.append(stream[i+1])
            stream_with_chords[-1].offset = stream[i+1].offset
                
    return stream_with_chords


# Small Demo
if __name__ == "__main__":
    
    import os

    path = os.getcwd()[:-4]
    # Reads a MIDI file
    bach = ms.converter.parse(path + "bach_846.mid")
    # Encodes the first hand of bach
    notes_encoded = encode(bach[0])
    print("\nSee below an example of an encoded notes\n")
    print(notes_encoded[10])
    print("\nOur network will learn the pattern that a sequence of these vectors follow in time,",
        "although we will probably not use the last component (tempo)")
    # Decodes it
    bach_decoded = decode(notes_encoded)
    # Saves and compares
    bach_decoded.write("midi", "bach_846-right_hand-decoded.mid")
    bach[0].write("midi", "bach_846-right_hand-original.mid")
