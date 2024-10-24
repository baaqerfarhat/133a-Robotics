import matplotlib.pyplot as plt
import numpy as np
import wave 
from numpy import linalg as LA


def save_data_to_wav(filename, wav, data):
    # round the data to int16 format for writing to .wav file
    data   = data.astype(np.int16)
    outwav = wave.open(filename, 'w')
    outwav.setparams(wav.getparams())
    outwav.setnchannels(1)
    # convert data to bytes for storage
    outwav.writeframes(data.tobytes())
    outwav.close()
    
def read_from_wav(filename):
    # the output: int16 numpy array
    obj  = wave.open(filename, 'rb')
    data = obj.readframes(-1)
    data = np.frombuffer(data, dtype='int16')
    obj.close()
    return data

fn = "sample.wav" # change to the correct location 
data = read_from_wav(fn)

## Your work here
#  Processing of data
data_processed = np.zeros(10) # replace this line by your processing steps...
##

# store data_processed into .wav file with parameters the same as "sample.wav"
wav = wave.open(fn, 'rb')
save_data_to_wav("output.wav", wav, data_processed)
wav.close()
 