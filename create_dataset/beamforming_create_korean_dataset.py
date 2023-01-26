
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
from tqdm import tqdm
from utils import preprocess_signal, cut_signals, draw_spectogram, DoughertyLogSpiral
import random
import math 
import csv
import pandas as pd
import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')

def read_mic_coords(filename, center_coord):
    data = pd.read_csv(filename)
    X = data['Dougherty_log_spiral'][1:]
    Y = data['Unnamed: 2'][1:]

    coords_list = []
    for x, y in zip(X, Y):
        coords_list.append([float(x)+center_coord[0], float(y)+center_coord[1]])
    
    coords_list = np.array([list(reversed(col)) for col in zip(*coords_list)])
    coords_list = coords_list[:, ::-1]
    return coords_list

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


def create_beamformed_data(signal1_path, source_1_gain, noise_path, noise_gain, save_mixed_path, filename):
    
    
    Fs = 16000
    absorption = 0.1
    max_order_sim = random.randint(0, 4)
    sigma2_n = 5e-7
    c = 343.0
    Lg_t = 0.100
    Lg = np.ceil(Lg_t * Fs)
    delay = 0.050
    fft_len = 1024
    n_mics = 12

    signal1, fs1 = sf.read(signal1_path)
    noise, fs3 = sf.read(noise_path)

    signal1 = preprocess_signal(fs1,Fs,signal1) * source_1_gain
    noise = preprocess_signal(fs3,Fs,noise) * noise_gain

    signal1, noise = set_signal_length([signal1, noise])

    sf.write(os.path.join(save_mixed_path, 's1', filename+'.wav'), signal1, Fs)
    sf.write(os.path.join(save_mixed_path, 'noise', filename+'.wav'), noise, Fs)

    room_dim = [random.randrange(5,20),random.randrange(5,20)]
    rmax = 0.15
    mic_center = [random.uniform(rmax, room_dim[0]/2), random.uniform(rmax, room_dim[1])]
    sig1_pos = [random.uniform(room_dim[0]/2, room_dim[0])-2, random.uniform(rmax, room_dim[1] - 2)]
    # sig2_pos = [sig1_pos[0], sig1_pos[1] + random.uniform(0.1, 0.5)]
    # noise_pos = [random.uniform(rmax, room_dim[0]/2), random.uniform(rmax, room_dim[1])]
    noise_pos = [sig1_pos[0]+ random.uniform(1, 2), sig1_pos[1] + random.uniform(1, 2)]

    room = pra.ShoeBox(
        room_dim,
        absorption=absorption,
        fs=Fs,
        max_order=max_order_sim,
        sigma2_awgn=sigma2_n,
        air_absorption = True
    )

    R = np.array(read_mic_coords("../../../sound_data/drone_test/mics.csv", mic_center))
    mics = pra.Beamformer(R, room.fs, N=fft_len, Lg=Lg)

    room.add_source(sig1_pos, delay=0., signal=signal1)
    room.add_source(noise_pos, delay=0., signal=noise)

    room.add_microphone_array(mics)
    room.compute_rir()
    room.simulate()
    
    sf.write(os.path.join(save_mixed_path, 'single_mic', filename+'.wav'), room.mic_array.signals[-1,:],  Fs)

    room.mic_array.rake_delay_and_sum_weights(room.sources[0][:1])#, None, R_n = sigma2_n * np.eye(mics.Lg * mics.M))


    fig, ax = room.plot(freq=[500, 1000, 2000, 4000, 8000], img_order=0)
    ax.legend(['500', '1000', '2000', '4000', '8000'])
    # fig.set_size_inches(20, 20)
    plt.savefig(os.path.join(save_mixed_path, 'figure', f'{filename}.png'), dpi=600)
    plt.close(fig)
    plt.cla()
    plt.clf()

    signal_das = room.mic_array.process(FD=False)
    signal_das = pra.normalize(signal_das, 16)
    signal_das = signal_das[-signal1.shape[0]:]

    sf.write(os.path.join(save_mixed_path, 'mix', filename+'.wav'), signal_das.astype(np.int16), Fs)

    f.write(f'{filename},{signal1_path},{source_1_gain},{noise_path},{noise_gain},{room_dim},{max_order_sim},{mic_center},{sig1_pos},{noise_pos}\n')

if __name__ == "__main__":


    source_base_path = r'D:\Project\sound_data\korean_voice\data\remote\PROJECT\AI학습데이터\KoreanSpeech\data\1.Training\2.원천데이터'
    noise_base_path = r'D:\Project\sound_data\LibriMix\data\wham_noise\tr'

    count = 0

    source_path_list = []
    noise_path_list = []

    for root, dirs, files in os.walk(source_base_path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                # f.write(f'{file_path}\n')
                source_path_list.append(file_path)
                count +=1

    print(count)

    
    count = 0
    for root, dirs, files in os.walk(noise_base_path):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                # f.write(f'{file_path}\n')
                noise_path_list.append(file_path)
                count +=1

    print(count)

    # in_csv_path = '../../sound_data/LibriMix/metadata/Libri2Mix/libri2mix_train-clean-360.csv'
    # base_data_path = '../../sound_data/LibriMix/data'
    save_mixed_path = r'G:\sound_data\custom_bf_20220504'

    

    os.makedirs(save_mixed_path, exist_ok=True)
    os.makedirs(os.path.join(save_mixed_path, 's1'), exist_ok=True)
    os.makedirs(os.path.join(save_mixed_path, 'noise'), exist_ok=True)
    os.makedirs(os.path.join(save_mixed_path, 'mix'), exist_ok=True)
    os.makedirs(os.path.join(save_mixed_path, 'single_mic'), exist_ok=True)
    os.makedirs(os.path.join(save_mixed_path, 'figure'), exist_ok=True)

    f = open(os.path.join(save_mixed_path, 'data_info.csv'), 'w')
    f.write(f'filename,signal1_path,source_1_gain,noise_path,noise_gain,room_size_x,room_size_y,max_order_rir,mic_center_coord_x,mic_center_coord_y,sig1_position_x,sig1_position_y,noise_pos_position_x,noise_pos_position_y\n')
    idx = 0
    for source_path in tqdm(source_path_list):
        # if idx > 23676:

        signal1_path = source_path
        source_1_gain = random.uniform(0.5, 1)
        noise_path = noise_path_list[random.randint(0, 59999)]
        noise_gain = random.uniform(0.7, 1.5)

        filename = f'{idx:0>7}'
        create_beamformed_data(signal1_path, source_1_gain, noise_path, noise_gain, save_mixed_path, filename)

        idx +=1

        # print(f'{idx}/{len(source_path_list)} complete...\r', end = '')
        
        # input('complete')

    f.close()