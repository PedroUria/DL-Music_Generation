# As I feared, with this approach we lose info. The hold dimension on the vectors 
# for the right and left hands gets mixed up and as a result the offsets get mixed up too. 
# It's a pitty, but I am pretty sure there is no way to get the exactly correct offsets 
# after encoding them this way, because even when differentiating between holds for left and right hands,
# there is no way to actually differentiate between notes for left and right hands.
# See the two files "multi-many-hot" on their names, and compare with the originals
# Still, let us try training the networks with this approach
# To do so, we need to tweak the decode function a little bit


import music21 as ms
import numpy as np
from encoder_decoder import encode


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
    if notes_encoded[0, -1]:  # If we encode and decode the left hand by separate
        stream.append(ms.tempo.MetronomeMark(number=int(notes_encoded[0, -1])))
    else:  # It will have tempo=0, so we need this.
        stream.append(ms.tempo.MetronomeMark(number=tempo))
    stream.append(ms.key.Key("C"))
    stream.append(ms.meter.TimeSignature())
    
    time_step = time_step
    offset = 0
    hold = {}  # Will store the duration of the hold for each note
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
            if p[-2]:  
                i = j + 1  
                dur += time_step
                while notes_encoded[i][-2]:  
                    dur += time_step
                    i += 1
            hold[87] = dur + offset  
            nt.duration = ms.duration.Duration(dur)
            stream.append(nt)
            stream[-1].offset = offset
        # For each note on the vector at time = offset
        for frequ_index in np.nonzero(notes_freq * p[:87])[0]:
            try:  # If the duration of the hold + offset is longer than the current offset
                if offset < hold[frequ_index]:
                    continue  # Do not append this note to the stream
            except:  # As it will be the same note appended on the first previous iteration
                pass
            # Gets the pitch for a note in p
            nt = ms.note.Note(notes_letters[int(frequ_index)])
            # Gets the duration
            dur = time_step
            if p[-2]:  # If we have a hold
                i = j + 1  # Move onto the next p vector
                dur += time_step
                while notes_encoded[i][-2]:  # And do so until the hold dissapears
                    dur += time_step
                    i += 1
            hold[frequ_index] = dur + offset  # Total duration of the hold, from the offset
            nt.duration = ms.duration.Duration(dur)
            # Appends to the stream
            stream.append(nt)
            # Sets the offset (need to do it here)
            stream[-1].offset = offset

        offset += time_step
                
    return stream


def multi_many_hot_encode(midi_file):
    
    """ Multi-many-hot-encodes two hands """

    hands = ms.converter.parse(midi_file)

    voice = False  # If there is more than one voice on
    for idx, nt in enumerate(hands[0]):  # the right hand (first part), just
        if type(nt) == ms.stream.Voice:  # takes the first voice
            voice = True
            break
    if voice:
        right_notes = encode(hands[0][idx])
    else:
        right_notes = encode(hands[0])
    voice = False  # And the same for the left hand
    for idx, nt in enumerate(hands[1]):
        if type(nt) == ms.stream.Voice:
            voice = True
            break
    if voice:
        left_notes = encode(hands[1][idx])
    else:
        left_notes = encode(hands[1])

    # Combines both encoded hands
    if right_notes.shape >= left_notes.shape:
        notes_combined = np.zeros(right_notes.shape)
        for idx, left_note in enumerate(left_notes):
            notes_combined[idx] = left_note
        notes_combined += right_notes
    else:
        notes_combined = np.zeros(left_notes.shape)
        for idx, right_note in enumerate(right_notes):
            notes_combined[idx] = right_note
        notes_combined += left_notes

    # In case there is a rest or a hold 
    # on both hands at the same time
    for idx, nt in enumerate(notes_combined):
        if nt[88] == 2:
            notes_combined[idx][88] = 1
        if nt[87] == 2:
            notes_combined[idx][87] = 1

    return notes_combined


if __name__ == "__main__":

    import os

    path = os.getcwd()[:-4]
    # Encodes both hands at the same time
    notes_combined = multi_many_hot_encode(path + "bach_846.mid")
    # Decodes the combined hands
    notes_combined_decoded = decode(notes_combined)
    # Saves the result as a MIDI file
    notes_combined_decoded.write("midi", "bach_right_left_multi-many-hot-encoded_decoded.mid")

    # Same with fur-elise
    notes_combined = multi_many_hot_encode(path + "elise.mid")
    notes_combined_decoded = decode(notes_combined)
    notes_combined_decoded.write("midi", "elise_right_left_multi-many-hot-encoded_decoded.mid")
