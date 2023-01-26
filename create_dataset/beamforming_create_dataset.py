
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra
# from pyroomacoustics.directivities import (
#     DirectivityPattern,
#     DirectionVector,
#     CardioidFamily,
# )
import soundfile as sf
import samplerate
from scipy.signal import butter, lfilter
import os
import librosa
import librosa.display
from utils import preprocess_signal, cut_signals, draw_spectogram, DoughertyLogSpiral
import random
import math 
import csv

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

def set_signal_length(signals):
    max_length = 0
    for signal in signals:
        if len(signal) > max_length:
            max_length = len(signal)

    outsigs = []
    for signal in signals:
        outsig = np.zeros(max_length)
        outsig[:len(signal)] = signal
        outsigs.append(outsig)

    return outsigs

def circular_3d_coords(center, radius, num, direction = "virtical"):
    
    list_coords = []

    if direction == "vertical":
        for i in range(num):
            list_coords.append([center[0], center[1] + radius*np.sin(2*i*np.pi/num), center[2] + radius*np.cos(2*i*np.pi/num)])

    elif direction == "horizontal":
        for i in range(num):
            list_coords.append([center[0]+ radius*np.sin(2*i*np.pi/num), center[1]+ radius*np.cos(2*i*np.pi/num), center[2] ])
    list_coords = [list(reversed(col)) for col in zip(*list_coords)]

    return np.array(list_coords)


def create_beamformed_data(signal1_path, source_1_gain, signal2_path, source_2_gain, noise_path, noise_gain, save_mixed_path, filename):
    

    Fs = 8000
    absorption = 0.1
    max_order_sim = 3
    sigma2_n = 5e-7
    c = 343.0
    Lg_t = 0.100
    Lg = np.ceil(Lg_t * Fs)
    delay = 0.050
    fft_len = 1024
    n_mics = 12

    signal1, fs1 = sf.read(signal1_path)
    signal2, fs2 = sf.read(signal2_path)
    noise, fs3 = sf.read(noise_path)

    total_gain = source_1_gain + source_2_gain + noise_gain
    signal1 = preprocess_signal(fs1,Fs,signal1) * source_1_gain
    signal2 = preprocess_signal(fs2,Fs,signal2) * source_2_gain
    noise = preprocess_signal(fs3,Fs,noise) * noise_gain

    signal1, signal2, noise = set_signal_length([signal1, signal2, noise])

    sf.write(os.path.join(save_mixed_path, 's1', filename+'.wav'), signal1, Fs)
    sf.write(os.path.join(save_mixed_path, 's2', filename+'.wav'), signal2, Fs)

    room_dim = [random.randrange(5,20),random.randrange(5,20)]
    rmax = 0.15
    mic_center = [random.uniform(rmax, room_dim[0]/2), random.uniform(rmax, room_dim[1])]
    sig1_pos = [random.uniform(room_dim[0]/2, room_dim[0]), random.uniform(rmax, room_dim[1] - 0.5)]
    sig2_pos = [sig1_pos[0], sig1_pos[1] + random.uniform(0.1, 0.5)]
    noise_pos = [random.uniform(rmax, room_dim[0]/2), random.uniform(rmax, room_dim[1])]

    room = pra.ShoeBox(
        room_dim,
        absorption=absorption,
        fs=Fs,
        max_order=max_order_sim,
        sigma2_awgn=sigma2_n,
        air_absorption = True
    )

    R = DoughertyLogSpiral(mic_center, rmax = rmax, rmin = 0.025, v = 87 *np.pi/180, n_mics = n_mics, direction = "2d")
    mics = pra.Beamformer(R, room.fs, N=fft_len, Lg=Lg)

    room.add_source(sig1_pos, delay=0., signal=signal1)
    room.add_source(sig2_pos, delay=0., signal=signal2)
    room.add_source(noise_pos, delay=0., signal=noise)

    room.add_microphone_array(mics)
    room.compute_rir()
    room.simulate()
    room.plot()
    plt.savefig()

    sf.write(os.path.join(save_mixed_path, 'single_mic', filename+'.wav'), pra.normalize(room.mic_array.signals[-1,:], 16).astype(np.int16),  Fs)

    room.mic_array.rake_delay_and_sum_weights(room.sources[0][:2], None, R_n = sigma2_n * np.eye(mics.Lg * mics.M))
    signal_das = room.mic_array.process(FD=False)
    signal_das = pra.normalize(signal_das, 16)
    sf.write(os.path.join(save_mixed_path, 'mix', filename+'.wav'), signal_das.astype(np.int16), Fs)

if __name__ == "__main__":
    in_csv_path = '../../sound_data/LibriMix/metadata/Libri2Mix/libri2mix_train-clean-360.csv'
    base_data_path = '../../sound_data/LibriMix/data'
    save_mixed_path = '../../sound_data/LibriMix/bf_custom'

    os.makedirs(save_mixed_path, exist_ok=True)
    os.makedirs(os.path.join(save_mixed_path, 's1'), exist_ok=True)
    os.makedirs(os.path.join(save_mixed_path, 'noise'), exist_ok=True)
    os.makedirs(os.path.join(save_mixed_path, 'mix'), exist_ok=True)
    os.makedirs(os.path.join(save_mixed_path, 'single_mic'), exist_ok=True)

    with open(in_csv_path, "r", encoding="UTF8") as f:
        rdr = csv.reader(f)
        for idx, line in enumerate(rdr):
            if idx == 0:
                continue

            mixture_ID = line[0]
            signal1_path = os.path.join(base_data_path, 'LibriSpeech', line[1])
            source_1_gain = float(line[2])
            signal2_path = os.path.join(base_data_path, 'LibriSpeech', line[3])
            source_2_gain = float(line[4])
            noise_path = os.path.join(base_data_path, 'wham_noise', line[5])
            noise_gain = float(line[6])

            create_beamformed_data(signal1_path, source_1_gain, signal2_path, source_2_gain, noise_path, noise_gain, save_mixed_path, mixture_ID)

            # print(f'{idx}/50801 complete...\r', end = '')
            
            # input('complete')