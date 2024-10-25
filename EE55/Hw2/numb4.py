import matplotlib.pyplot as plt
import numpy as np
import wave

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
data = read_from_wav(fn)

X = np.fft.fft(data)
b_values = [5000, 10000, 40000]

plot_dir = r"C:\Users\farha\Desktop\Caltech\2024-2025\EE55\Hw2\plots"

T = len(data)
fs = 48000
time = np.arange(T) / fs
frequencies = np.fft.fftfreq(T, d=1/fs)

for b in b_values:
    X_proj = np.zeros_like(X)
    X_proj[:b] = X[:b] 

    x_proj = np.fft.ifft(X_proj).real
    mse = np.mean((data - x_proj) ** 2)
    print(f"MSE for b={b}: {mse}")

    plt.figure()
    plt.plot(time, data, label='Original Signal')
    plt.plot(time, x_proj, label=f'Projected Signal (b={b})')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(f'Original vs Projected Signal (b={b}) in Time Domain')
    plt.legend()
    plt.savefig(f"{plot_dir}/time_domain_b_{b}.png")
    plt.close()

    plt.figure()
    plt.plot(frequencies[:T // 2], np.abs(X[:T // 2]), label='Original Signal')
    plt.plot(frequencies[:T // 2], np.abs(X_proj[:T // 2]), label=f'Projected Signal (b={b})')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title(f'Fourier Transform Magnitude (b={b})')
    plt.legend()
    plt.savefig(f"{plot_dir}/freq_domain_b_{b}.png")
    plt.close()

    output_wav_filename = f"{plot_dir}/projected_b_{b}.wav"
    wav = wave.open(fn, 'rb')
    save_data_to_wav(output_wav_filename, wav, x_proj)
    wav.close()

print("Processing complete. Plots and WAV files have been saved.")
