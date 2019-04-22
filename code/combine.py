import music21 as ms


def combine(left, right, filepath='test.mid'):
    """
    Input: left - the stream of notes played by the 
                pianist with his left hand
            right - the stream of notes played by the 
                pianist with his right hand
            filepath - path to the file in which o/p
                midi stream is to be stored
    Output: N/A

    The combines the streams from the left stream and right 
    stream and outputs a single stream object
    """
    sc = ms.stream.Stream()
    sc.append(right)
    sc.append(left)
    left.offset = 0.0
    sc.write("midi", filepath)
