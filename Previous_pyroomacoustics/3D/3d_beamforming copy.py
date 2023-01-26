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


if __name__ == "__main__":

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
    # sigma2_n = None
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

    fs1, signal1 = wavfile.read("./dataset/original/dji_marvicpro.wav")


    fs = fs1
    # Fs = fs/2
    Fs = 8000
    signal1 = preprocess_signal(fs1,Fs,signal1)

    room_dim=[5000,5000,100]
    room = pra.ShoeBox(
        room_dim,
        absorption=absorption,
        fs=Fs,
        max_order=max_order_sim,
        sigma2_awgn=sigma2_n,
        air_absorption = True
    )


    sig1_pos = [2500, 2504, 1.5]
    room.add_source(sig1_pos, delay=0., signal=signal1)

    mic_center = [2500, 2500, 1.5]
    mic_radius = 0.05
    fft_len = 1024
    angle = 90
    R = DoughertyLogSpiral(mic_center, rmax = 0.15, rmin = 0.025, v = 87 *np.pi/180, n_mics = 112, direction = 'vertical')
    
    mics = pra.Beamformer(R, room.fs, N=fft_len, Lg=Lg)
    room.add_microphone_array(mics)
    room.compute_rir()
    room.simulate()
    room.plot()
    plt.show()

    
    sf.write('./results/3d/beamforming/drone/all_mix.wav', room.mic_array.signals[-1,:].astype(np.int16),  fs)

    # room.mic_array.rake_perceptual_filters(room.sources[0][:2], None, R_n = sigma2_n * np.eye(mics.Lg * mics.M))
    room.mic_array.rake_perceptual_filters(room.sources[0][:2], None, R_n = sigma2_n * np.eye(mics.Lg * mics.M), delay = delay)
    signal_das = room.mic_array.process(FD=False)
    print(type(signal_das))
    print(type(signal_das[0]))
    print(np.max(signal_das))
    print(np.min(signal_das))
    signal_das = np.array(signal_das/4, dtype=np.int16)

    print(type(signal_das))
    print(type(signal_das[0]))
    print(np.max(signal_das))
    print(np.min(signal_das))
    # signal_das = pra.normalize(signal_das, 16)
    sf.write(f'./results/3d/beamforming/drone/drone_test_source_0.wav', signal_das, fs)
    
    # room.mic_array.rake_perceptual_filters(room.sources[1][:2], room.sources[0][:2], sigma2_n * np.eye(mics.Lg * mics.M))
    # signal_das = room.mic_array.process(FD=False)
    # signal_das = pra.normalize(signal_das, 16)
    # sf.write(f'./results/3d/beamforming/perceptual_source_1.wav', signal_das.astype(np.int16), fs)
