from turtle import shape
from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)
import soundfile as sf
import samplerate
from scipy import signal
from scipy.signal import butter, lfilter
import os
import librosa
import librosa.display
from utils import preprocess_signal, cut_signals, draw_spectogram, DoughertyLogSpiral

def preprocess_data(signal, Fs):
    np.array(signal, dtype=float)
    signal = pra.highpass(signal, Fs)
    signal = pra.normalize(signal)

    if np.shape(signal)[1]:
        signal = np.squeeze(signal[:,0])
    return signal

def resample(ori_rate,new_rate,signal):
    fs_ratio = new_rate / float(ori_rate)
    signal = samplerate.resample(signal, fs_ratio, "sinc_best")
    return signal


def circular_3d_coords(center, radius, num, direction = 'virtical'):
    
    list_coords = []

    if direction == 'vertical':
        for i in range(num):
            list_coords.append([center[0], center[1] + radius*np.sin(2*i*np.pi/num), center[2] + radius*np.cos(2*i*np.pi/num)])

    elif direction == 'horizontal':
        for i in range(num):
            list_coords.append([center[0]+ radius*np.sin(2*i*np.pi/num), center[1]+ radius*np.cos(2*i*np.pi/num), center[2] ])
    list_coords = [list(reversed(col)) for col in zip(*list_coords)]

    return np.array(list_coords)


def duplicate_sound(signal, times = 3):
    signal_duplicated = np.zeros(len(signal)*times,dtype=type(signal[0]))

    for i in range(times):
        signal_duplicated[len(signal) * i :len(signal)*(i+1)] = signal

    return signal_duplicated

def rechannel(signal):
    if len(signal.shape) == 2:
        signal = signal[...,0]

    return signal


def main():
    '''
        s1, s1 ,s2 ,n1  
        sum + rir 
        rakeMVDR, delay and sum, rake pertual MVDR  -- 스펙트로그램까지 
    '''
    # Spectrogram figure properties
    figsize = (15, 7)  # figure size
    fft_size = 512  # fft size for analysis
    fft_hop = 8  # hop between analysis frame
    fft_zp = 512  # zero padding
    analysis_window = pra.hann(fft_size)
    t_cut = 0.83  # length in [s] to remove at end of signal (no sound)    


    # Some simulation parameters
    Fs = 8000
    absorption = 0.1
    max_order_sim = 0
    sigma2_n = 5e-7
    c = 343.

    # Microphone array design parameters
    mic_n = 8  # number of microphones
    d = 0.08  # distance between microphones
    phi = 0.0  # angle from horizontal
    max_order_design = 1  # maximum image generation used in design
    shape = "Linear"  # array shape
    Lg_t = 0.100  # Filter size in seconds
    Lg = np.ceil(Lg_t * Fs)  # Filter size in samples
    delay = 0.050  # Beamformer delay in seconds

    # Define the FFT length
    N = 1024
    nfft = 256

    #Define two signal and one noise
    path = os.path.dirname(__file__) 

    fs1, signal1 = wavfile.read("./dataset/original/Machine.wav")
    
    
    signal1 = rechannel(signal1)
    signal1 = pra.normalize(signal1, 16)

    fs2, signal2 = wavfile.read("./dataset/original/glitchy-noise-fx_128bpm_F.wav")
    # print(signal2)
    # print(signal1[0])
    
    signal2 = rechannel(signal2)
    signal2 = pra.normalize(signal2, 16)

    print(np.max(signal2), np.min(signal2))

    fs = fs2
    signal1 = preprocess_signal(fs1,fs,signal1)
    signal2_duplicated = duplicate_sound(signal2, times = 10)

    signals = cut_signals([signal1, signal2_duplicated])

    a = 0.8

    signal_sum = signals[0]* a +  signals[1]*(1-a)

    print(np.max(signal2_duplicated), np.min(signal2_duplicated))
    sf.write("./dataset/original/hf_noise_duplicated.wav", signal2_duplicated.astype(np.int16), fs)
    sf.write("./dataset/original/Machine_noised.wav", signal_sum.astype(np.int16), fs)
  

if __name__ == "__main__":
    path_list = ["./dataset/original/temp/1_est2.wav", "./dataset/original/temp/1_est1.wav", "./dataset/original/temp/1.wav", "./dataset/original/temp/1_est3.wav"]

    for path in path_list:
        fs1, signal1 = wavfile.read(path)

        signal1 = signal1[2*fs1:8*fs1]

        sf.write(f'{path[:-4]}_cropped.wav', signal1, fs1)
    
    print(signal1.shape)
    # signal1 = rechannel(signal1)
    # signal1 = pra.normalize(signal1, 16)

    # # cropped_sig = signal1[fs1*16:fs1*23]

    # sf.write("./dataset/original/Traffic_Noise.wav", signal1.astype(np.int16), fs1)