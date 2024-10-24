import matplotlib.pyplot as plt
import numpy as np
import wave
import os
from numpy import linalg as LA

def save_data_to_wav(filename, wav, data):
    data = data.astype(np.int16)
    outwav = wave.open(filename, 'w')
    outwav.setparams(wav.getparams())
    outwav.setnchannels(1)
    outwav.writeframes(data.tobytes())
    outwav.close()

def read_from_wav(filename):
    obj = wave.open(filename, 'rb')
    data = obj.readframes(-1)
    data = np.frombuffer(data, dtype='int16')
    obj.close()
    return data

fn = r"C:\Users\farha\Desktop\Caltech\2024-2025\EE55\Hw2\sample.wav"
plot_dir = r"C:\Users\farha\Desktop\Caltech\2024-2025\EE55\Hw2\plots"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

data = read_from_wav(fn)
d = len(data)
X = np.fft.fft(data)

def projection_Omega_b(X, b):
    mask = np.zeros_like(X, dtype=bool)
    mask[:b] = True
    mask[-b:] = True
    return X * mask

b_values = [5000, 10000, 40000]

for b in b_values:
    X_proj = projection_Omega_b(X, b)
    x_proj = np.fft.ifft(X_proj).real

    plt.figure(figsize=(12, 6))
    plt.plot(np.abs(x_proj))
    plt.title(f"Time domain projection for b = {b}")
    plt.xlabel("Time index")
    plt.ylabel("Amplitude")
    plt.savefig(os.path.join(plot_dir, f"time_domain_projection_b_{b}.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(np.abs(X_proj))
    plt.title(f"Frequency domain projection for b = {b}")
    plt.xlabel("Frequency index")
    plt.ylabel("Magnitude")
    plt.savefig(os.path.join(plot_dir, f"frequency_domain_projection_b_{b}.png"))
    plt.close()

    mse = np.mean(np.abs(data - x_proj)**2)
    print(f"MSE for b = {b}: {mse}")

    wav = wave.open(fn, 'rb')
    save_data_to_wav(f"output_b_{b}.wav", wav, x_proj)
    wav.close()
