import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
from scipy.io import wavfile
import soundfile as sf
import samplerate
import librosa
import librosa.display

def rechannel(signal):

    if len(signal.shape) == 2:
        signal = signal[...,0]

    return signal

def resample(ori_rate, new_rate, signal):
    fs_ratio = new_rate / float(ori_rate)
    signal = samplerate.resample(signal, fs_ratio, "sinc_best")
    return signal


def preprocess_signal(ori_rate, new_rate, signal):
    signal = rechannel(signal)
    signal = resample(ori_rate,new_rate,signal)
    # signal = pra.normalize(signal)
    return signal

def cut_signals(signals):

    length = np.inf

    for signal in signals:
        if signal.shape[0] < length:
            length = signal.shape[0]

    length = int(length/2)
    cut_signals = [signal[:length] for signal in signals]

    return cut_signals

def lowpass_filtering(signal):

    h_len = 50
    h = np.ones(h_len)
    h /= np.linalg.norm(h)

    # stft parameters
    fft_len = 512
    block_size = fft_len - h_len + 1  # make sure the FFT size is a power of 2
    hop = block_size // 2  # half overlap
    window = pra.hann(block_size, flag='asymmetric', length='full') 

    # Create the STFT object + set filter and appropriate zero-padding
    stft = pra.transform.STFT(block_size, hop=hop, analysis_window=window, channels=1)
    stft.set_filter(h, zb=h.shape[0] - 1)

    fs, signal = wavfile.read("arctic_a0010.wav")

    processed_audio = np.zeros(signal.shape)
    n = 0
    while  signal.shape[0] - n > hop:

        stft.analysis(signal[n:n+hop,])
        stft.process()  # apply the filter
        processed_audio[n:n+hop,] = stft.synthesis()
        n += hop
    
    return processed_audio

def draw_spectogram(test_data, Fs, fft_size, fft_hop, analysis_window):

    S = librosa.feature.melspectrogram(y=test_data, sr=Fs,n_fft=fft_size,hop_length=fft_hop, n_mels=64,window=analysis_window) 
    
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=Fs, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    # plt.savefig(path + "/output_samples/spectrograms_all_45ms.png", dpi=150)
    # plt.show()


def cot(x):
    return 1/np.tan(x)

def DoughertyLogSpiral(center, rmax, rmin, v, n_mics, direction = 'vertical'):

    list_coords = []

    l_max = rmin * np.sqrt(1 + cot(v)**2) / (cot(v)) * (rmax/rmin - 1)
    l_n = [i/(n_mics-1) * l_max for i in range(n_mics)] 
    # l_mx = rmin * np.sqrt(1 + cot(v)**2) / (cot(v) * (rmax/rmin - 1))

    Theta = [np.log(1 + cot(v) * x / (rmin*np.sqrt(1 + cot(v)**2)))/cot(v) for x in l_n]

    R = [rmin * np.e**(cot(v)*x) for x in Theta]

    X = [ r * np.cos(theta) for theta, r in zip(Theta, R)]
    Y = [ r * np.sin(theta) for theta, r in zip(Theta, R)]

    
    if direction == 'vertical':
        for x, y, in zip(X, Y):
            list_coords.append([center[0], center[1] + x, center[2] + y])

    elif direction == 'horizontal':
        for x, y, in zip(X, Y):
            list_coords.append([center[0]+ x, center[1]+ y, center[2]])

    if direction == '2d':
        for x, y, in zip(X, Y):
            list_coords.append([center[0]+ x, center[1]+ y])

    list_coords = [list(reversed(col)) for col in zip(*list_coords)]

    return np.array(list_coords)


def cut_signals(signals):

    length = np.inf

    for signal in signals:
        if signal.shape[0] < length:
            length = signal.shape[0]

    # length = int(length/2)
    cut_signals = [signal[:length] for signal in signals]

    return cut_signals