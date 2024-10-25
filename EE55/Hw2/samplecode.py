import matplotlib.pyplot as plt
import numpy as np
import wave

# Function to save processed data to a .wav file
def save_data_to_wav(filename, wav, data):
    data = data.astype(np.int16)  # Ensure data is in int16 format
    outwav = wave.open(filename, 'w')
    outwav.setparams(wav.getparams())
    outwav.setnchannels(1)
    outwav.writeframes(data.tobytes())  # Convert data to bytes for writing
    outwav.close()

# Function to read data from a .wav file
def read_from_wav(filename):
    obj = wave.open(filename, 'rb')
    data = obj.readframes(-1)
    data = np.frombuffer(data, dtype='int16')
    obj.close()
    return data

# Main workflow
fn = r"C:\Users\farha\Desktop\Caltech\2024-2025\EE55\Hw2\sample.wav"
data = read_from_wav(fn)

# Fourier transform of the signal using numpy's FFT (fast Fourier transform)
X = np.fft.fft(data)

# Define different b values for low-pass filtering
b_values = [5000, 10000, 40000]

# Directory for saving plots
plot_dir = r"C:\Users\farha\Desktop\Caltech\2024-2025\EE55\Hw2\plots"

# Parameters for the audio signal
T = len(data)
fs = 48000
time = np.arange(T) / fs

for b in b_values:
    # Apply low-pass filtering: retain only the first b Fourier coefficients
    X_proj = np.zeros_like(X)
    X_proj[:b] = X[:b]  # Retain only the first b frequencies (low-pass)

    # Inverse Fourier transform to get back the time-domain signal
    x_proj = np.fft.ifft(X_proj).real

    # Compute the Mean Squared Error
    mse = np.mean((data - x_proj) ** 2)
    print(f"MSE for b={b}: {mse}")

    # Plot original vs projected signal in time domain
    plt.figure()
    plt.plot(time, data, label='Original Signal')
    plt.plot(time, x_proj, label=f'Projected Signal (b={b})')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(f'Original vs Projected Signal (b={b})')
    plt.legend()
    plt.savefig(f"{plot_dir}/time_domain_b_{b}.png")
    plt.close()

    # Save projected signal to WAV file
    output_wav_filename = f"{plot_dir}/projected_b_{b}.wav"
    wav = wave.open(fn, 'rb')
    save_data_to_wav(output_wav_filename, wav, x_proj)
    wav.close()

print("Processing complete. Plots and WAV files have been saved.")
