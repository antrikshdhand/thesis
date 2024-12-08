import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, chirp, windows
import math
import random

# Generate a sine wave with random frequency
def generate_sine_wave(duration, fs, amplitude=1.0, freq=None):
    if freq is None:
        freq = random.uniform(50, 1000)  # Random frequency between 100 Hz and 1000 Hz
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * freq * t)
    return t, sine_wave, freq

# Add Gaussian noise with random standard deviation
def add_gaussian_noise(signal, std=None):
    if std is None:
        std = random.uniform(2, 7) 
    noise = np.random.normal(0, std, signal.shape)
    return signal + noise, std

# Create a spectrogram using scipy's spectrogram function
def create_spectrogram(signal, fs, win, hop, nfft, power=False, T=False):
    f, t, Sxx = spectrogram(signal, fs=fs, window=win, noverlap=hop, nfft=nfft, mode='magnitude')
    if power:
        Sxx = 10 * np.log10(Sxx + 1e-8)
    if T:
        Sxx = Sxx.T
    return f, t, Sxx

# Plot spectrogram with time on y-axis and frequency on x-axis
def plot_spectrogram(f, t, Sxx, power=False, title="Spectrogram"):
    plt.figure(figsize=(10, 6))
    if power:
        plt.pcolormesh(f, t, Sxx.T + 1e-10, shading='auto')
    else:
        plt.pcolormesh(f, t, 10 * np.log10(Sxx.T + 1e-10), shading='auto')
    plt.colorbar(label='Intensity (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Time (s)')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    fs = 5000  # Sampling frequency
    duration = 3.0  # Duration in seconds
    windowLengthSec = 0.04
    windowLengthSeg = int(windowLengthSec * fs)  # Number of samples per window
    windowLengthSegRounded = 2 ** math.ceil(math.log2(windowLengthSeg))  # Next power of 2
    window = windows.hann(windowLengthSegRounded)  # Hann window
    noverlap = int(0.75 * windowLengthSegRounded)  # Overlap between windows
    nfft = 1024  # FFT size

    # Generate random sine wave
    _, sine_wave, freq = generate_sine_wave(duration, fs)
    print(f"Generated sine wave with frequency: {freq:.2f} Hz")

    # Add random noise to sine wave
    noisy_sine, sine_std = add_gaussian_noise(sine_wave)
    print(f"Added Gaussian noise to sine wave with stddev: {sine_std:.2f}")

    # Create spectrograms
    f, t, clean_spectrogram = create_spectrogram(sine_wave, fs, window, noverlap, nfft)
    _, _, noisy_spectrogram = create_spectrogram(noisy_sine, fs, window, noverlap, nfft)

    # Plot clean and noisy spectrograms
    plot_spectrogram(f, t, clean_spectrogram, title="Clean Sine Wave Spectrogram")
    plot_spectrogram(f, t, noisy_spectrogram, title="Noisy Sine Wave Spectrogram")

if __name__ == "__main__":
    main()
