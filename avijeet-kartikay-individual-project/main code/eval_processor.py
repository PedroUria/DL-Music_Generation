def eva(notes_decoded, part, accs=[], tempo_right=None, tempo_right_orig=None):
    tempo_en_dec = tempo_right
    tempo_orig = tempo_right_orig
    for nt in notes_decoded:
        if type(nt) == ms.tempo.MetronomeMark:
            tempo_en_dec = float(nt.number)
            break
    for nt in part:
        if type(nt) == ms.tempo.MetronomeMark:
            tempo_orig = float(nt.number)
            break
    li_decoded, li_orig = [], []
    for nt in notes_decoded:
        if type(nt) == ms.note.Note or type(nt) == ms.note.Rest: # or type(nt) == ms.chord.Chord:  # TODO
            li_decoded.append(nt)
    for nt in part:
        if type(nt) == ms.note.Note or type(nt) == ms.note.Rest: # or type(nt) == ms.chord.Chord:  # TODO
            li_orig.append(nt)
    s = (tempo_en_dec == tempo_orig)*1
    l = 1
    idx = 0
    for nt_enc_dec, nt_real in zip(li_decoded, li_orig):
        if type(nt_enc_dec) == ms.note.Note:
            if (nt_enc_dec.pitch.frequency, nt_enc_dec.offset, nt_enc_dec.duration.quarterLength) == (
                nt_real.pitch.frequency, nt_real.offset, nt_real.duration.quarterLength):
                s += 1
                l += 1
            else:
                #print("index", idx)
                #print(nt_enc_dec, nt_real)
                #print(nt_enc_dec.pitch.frequency == nt_real.pitch.frequency)
                #print(nt_enc_dec.offset)
                #print(nt_real.offset)
                #print(nt_enc_dec.offset == nt_real.offset)
                #print(nt_enc_dec.duration.quarterLength == nt_real.duration.quarterLength)
                #print(nt_enc_dec.duration.quarterLength, nt_real.duration.quarterLength)
                #break
                l += 1
        if type(nt_enc_dec) == ms.note.Rest:
            if (nt_enc_dec.offset, nt_enc_dec.duration.quarterLength) == (
                nt_real.offset, nt_real.duration.quarterLength):
                s += 1
                l += 1
            else:
                l += 1
                print(idx)
                print(nt_enc_dec, nt_real)
                break
        if type(nt_real) == ms.chord.Chord:
            if nt_real.pitches == nt_enc_dec.pitches:
                s += 1
                l += 1
            else:
                l += 1
        idx += 1
    accs.append(s/l)
    return tempo_en_dec, tempo_orig



def evaluate_enc_dec(midi_file):
    
    piece = ms.converter.parse(midi_file)
    if len(piece) > 1:
        accs = []
        tempo_right, tempo_right_orig = None, None
        for part in piece[:2]:
            #print(part)
            #print("\n-------\n")
            voices = False
            for idx, nt in enumerate(part):
                if type(nt) == ms.stream.Voice:
                    print("!")
                    voices = True
                    break
            if voices:
                notes_encoded = encode(part[idx])
                notes_decoded = decode(notes_encoded)
                tempo_right, tempo_right_orig = eva(notes_decoded, part[idx], accs, tempo_right=tempo_right, tempo_right_orig=tempo_right_orig)
                #print(accs)
                # This breaks my encoding function...
                #for idx1, nt in enumerate(part[idx+1:]):
                #    if type(nt) == ms.stream.Voice:
                #        break
                #part[idx+1].show("text")
                #notes_encoded = encode(part[idx+1+idx1])
                #notes_decoded = decode(notes_encoded)
                #tempo_right, tempo_right_orig = eva(notes_decoded, part, accs, tempo_right=tempo_right, tempo_right_orig=tempo_right_orig)
            else:
                #part[:10].show("text")
                #part[:10].show("text")
                notes_encoded = encode(part)
                notes_decoded = decode(notes_encoded)
                tempo_right, tempo_right_orig = eva(notes_decoded, part, accs, tempo_right=tempo_right, tempo_right_orig=tempo_right_orig)
    else:
        pass
        
    return sum(accs)/len(accs)


# Small Demo:
if __name__ == "__main__":

    import os
    from encoder_decoder import *

    path = os.getcwd()[:-4]
    print(evaluate_enc_dec(path + "bach_846.mid"))

