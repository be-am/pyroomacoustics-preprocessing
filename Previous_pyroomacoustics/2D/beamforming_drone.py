
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


def cot(x):
    return 1/np.tan(x)

def DoughertyLogSpiral(center, rmax, rmin, v, n_mics, direction = "vertical"):

    list_coords = []

    l_max = rmin * np.sqrt(1 + cot(v)**2) / (cot(v)) * (rmax/rmin - 1)
    l_n = [i/(n_mics-1) * l_max for i in range(n_mics)] 
    # l_mx = rmin * np.sqrt(1 + cot(v)**2) / (cot(v) * (rmax/rmin - 1))

    Theta = [np.log(1 + cot(v) * x / (rmin*np.sqrt(1 + cot(v)**2)))/cot(v) for x in l_n]

    R = [rmin * np.e**(cot(v)*x) for x in Theta]

    X = [ r * np.cos(theta) for theta, r in zip(Theta, R)]
    Y = [ r * np.sin(theta) for theta, r in zip(Theta, R)]

    
    if direction == "vertical":
        for x, y, in zip(X, Y):
            list_coords.append([center[0], center[1] + x, center[2] + y])

    elif direction == "horizontal":
        for x, y, in zip(X, Y):
            list_coords.append([center[0]+ x, center[1]+ y, center[2]])

    if direction == "2d":
        for x, y, in zip(X, Y):
            list_coords.append([center[0]+ x, center[1]+ y])

    list_coords = [list(reversed(col)) for col in zip(*list_coords)]

    return np.array(list_coords)

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


if __name__ == "__main__":
 
    Fs = 8000
    absorption = 0.1
    max_order_sim = 0
    sigma2_n = 5e-7
    c = 343.0
    Lg_t = 0.100
    Lg = np.ceil(Lg_t * Fs)
    delay = 0.050

    fs1, signal1 = wavfile.read("./dataset/original/dji_marvicpro.wav")
    fs2, signal2 = wavfile.read("./dataset/original/cafe.wav")

    signal1 = preprocess_signal(fs1,Fs,signal1)
    signal2 = preprocess_signal(fs2,Fs,signal2)

    signal1, signal2 = cut_signals([signal1, signal2])

    distance_list = [10, 50, 100, 500, 1000, 2000, 4000]
    
    for distance in distance_list:
        room_dim=[5000,5000]
        room = pra.ShoeBox(
            room_dim,
            absorption=absorption,
            fs=Fs,
            max_order=max_order_sim,
            sigma2_awgn=sigma2_n,
            air_absorption = True
        )

        mic_center = [2500, 100]
        fft_len = 1024
        n_mics = 12
        R = DoughertyLogSpiral(mic_center, rmax = 0.15, rmin = 0.025, v = 87 *np.pi/180, n_mics = n_mics, direction = "2d")
        mics = pra.Beamformer(R, room.fs, N=fft_len, Lg=Lg)

        sig1_pos = [2500, mic_center[1] + distance]
        room.add_source(sig1_pos, delay=0., signal=signal1)

        # sig2_pos = [2500 + 3, mic_center[1]]
        # room.add_source(sig2_pos, delay=0., signal=signal2)


        room.add_microphone_array(mics)
        room.compute_rir()
        room.simulate()
        # room.plot()
        # plt.show()
        
        sf.write(f"./results/2d/beamforming/drone/all_mix_{distance}m.wav", room.mic_array.signals[-1,:].astype(np.int16),  Fs)

        # room.mic_array.rake_perceptual_filters(room.sources[0][:2], None, R_n = sigma2_n * np.eye(mics.Lg * mics.M), delay = delay)
        # room.mic_array.rake_perceptual_filters(room.sources[0][:2], None, R_n = sigma2_n * np.eye(mics.Lg * mics.M), delay = delay)
        room.mic_array.rake_delay_and_sum_weights(room.sources[0][:1], None, R_n = sigma2_n * np.eye(mics.Lg * mics.M))

        # fig, ax = room.plot(freq=[500, 1000, 2000, 3000, 4000, 5000, 6000], img_order=0)
        # ax.legend(["500", "1000", "2000", "4000", "5000", "6000"])
        # fig.set_size_inches(20, 8)
        # plt.show()

        signal_das = room.mic_array.process(FD=False)
        print(f'distance = {distance:10f}')
        print(f'Before converting to original output signal: max = {np.max(signal_das)}, min = {np.min(signal_das)}')
        
        # signal_das = np.array(signal_das/4, dtype=np.int16)
        signal_das = pra.normalize(signal_das, 16)
        # signal_das = signal_das/4
        print(f'After dividing output signal by 2^2 : max = {np.max(signal_das)}, min = {np.min(signal_das)}')

        rms = np.sqrt(np.mean(signal_das**2))
        energy = np.mean(signal_das ** 2, axis=-1)

        print(f"sig_max = {np.max(signal_das):10f}, sig_min = {np.min(signal_das):10f}, rms = {rms:10f}, energy = {energy}")
        print('')
        sf.write(f"./results/2d/beamforming/drone/drone_test_source_0_{distance}m.wav", signal_das, Fs)
        
        # room.mic_array.rake_perceptual_filters(room.sources[1][:2], room.sources[0][:2], sigma2_n * np.eye(mics.Lg * mics.M))
        # signal_das = room.mic_array.process(FD=False)
        # signal_das = pra.normalize(signal_das, 16)
        # sf.write(f"./results/3d/beamforming/perceptual_source_1.wav", signal_das.astype(np.int16), fs)
