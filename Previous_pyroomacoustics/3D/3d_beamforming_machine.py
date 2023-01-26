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



    #Define two signal and one noise
    path = os.path.dirname(__file__) 

    fs1, signal1 = wavfile.read("./dataset/original/Machine2.wav")
    fs2, signal2 = wavfile.read("./dataset/original/Machine.wav")
    fs3, signal3 = wavfile.read("./dataset/original/Machine.wav")
    fs4, signal4 = wavfile.read("./dataset/original/Machine.wav")

    # Some simulation parameters
    Fs = 8000
    absorption = 0.1
    max_order_sim = 0
    sigma2_n = 5e-7
    c = 343.

    # Microphone array design parameters
    mic_n = 10  # number of microphones
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

    
    # signal1 = signal1[...,0]
    # signal2 = signal2[...,0]
    # signal3 = signal3[...,0]

    signal1 = preprocess_signal(fs1,Fs,signal1)
    signal2 = preprocess_signal(fs2,Fs,signal2)
    signal3 = preprocess_signal(fs3,Fs,signal3)
    signal4 = preprocess_signal(fs4,Fs,signal4)

    signals = cut_signals([signal1, signal2, signal3, signal4])

    # Create a 10X5X5 metres shoe box room
    room_dim=[5,10,5]
    room = pra.ShoeBox(
        room_dim,
        absorption=absorption,
        fs=Fs,
        max_order=max_order_sim,
        sigma2_awgn=sigma2_n,
    )

    # 30 degrees
    source1_loc = np.array([2.5, 2, 1.5])
    source2_loc = np.array([2.5, 4, 1.5])
    source3_loc = np.array([2.5, 6, 1.5])
    source4_loc = np.array([2.5, 8, 1.5])
    # interferer = np.array([3.5, 5, 1.5])

    # 10 degrees
    # source1 = np.array([5.25, 2.32, 1.5])
    # source2 = np.array([5.25, 2.85, 1.5])
    # source3 = np.array([5.25, 3.38, 1.5])
    # interferer = np.array([3.5, 5, 1.5])

    # # 5 degrees
    # source1 = np.array([5.25, 2.59, 1.5])
    # source2 = np.array([5.25, 2.85, 1.5])
    # source3 = np.array([5.25, 3.11, 1.5])
    # interferer = np.array([3.5, 5, 1.5])


    # sig1_pos = [1, 1, 1.5]
    # sig2_pos = [1, 5, 1.5]
    # noise_pos = [3,1,4]

    # room.add_source(noise_pos,signal=noise,delay=0)
    # room.plot()
    # plt.show()
    room.add_source(source1_loc, delay=0., signal=signals[0])
    # room.add_source(source2_loc, delay=0., signal=signals[1])
    # room.add_source(source3_loc, delay=0., signal=signals[2])
    room.add_source(source4_loc, delay=0., signal=signals[3])

    # mic_center = np.array([8, 3, 1])
    mic_center = [1, 5, 1.5]

    # microphone array radius
    mic_radius = 0.05
    fft_len = 1024
    R = circular_3d_coords(mic_center, mic_radius, mic_n, 'vertical')
    angle = 90
    # pattern = DirectivityPattern.CARDIOID
    # directivity = CardioidFamily(
    #     orientation=DirectionVector(azimuth=angle, colatitude=90, degrees=True), 
    #     pattern_enum=pattern
    # )
    # R = DoughertyLogSpiral(mic_center, rmax = 0.15, rmin = 0.025, v = 87 *np.pi/180, n_mics = 112, direction = 'vertical')
    
    # mics = pra.MicrophoneArray(R, room.fs, directivity=directivity)
    mics = pra.Beamformer(R, room.fs, N=fft_len, Lg=Lg)
    # mics = pra.MicrophoneArray(R, fs=room.fs, N=N, Lg=Lg)
    room.add_microphone_array(mics)
    # room.compute_rir()
    room.simulate()
    room.plot()
    plt.show()


    bf_algoritms = ['das', 'mvdr', 'perceptual']
    bf_algoritms = ['das', 'mvdr']
    algoritm = 'mvdr'
    
    sf.write('./results/3d/beamforming/20220315_all_mix.wav', room.mic_array.signals[-1,:].astype(np.int16),  Fs)

    plot = False
    for i in range(2):
        # if i != 3:
        #     room.mic_array.rake_mvdr_filters(room.sources[i][:2], room.sources[i+1][:2], sigma2_n * np.eye(mics.Lg * mics.M), delay=0.0, epsilon=0.005)
        # else:
        #     room.mic_array.rake_mvdr_filters(room.sources[3][:2], room.sources[2][:2], sigma2_n * np.eye(mics.Lg * mics.M), delay=0.0, epsilon=0.005)


        room.mic_array.rake_mvdr_filters(room.sources[i][:2], room.sources[1-i][:2], sigma2_n * np.eye(mics.Lg * mics.M), epsilon=0.005)

        signal_das = room.mic_array.process(FD=False)
        cut_fcs = [0,2000,4000,5000,6000, 7000]
        cut_fcs = [0] 

        for fc in cut_fcs:
            if fc != 0:
                signal_das_filtered = pra.highpass(signal_das, Fs, fc = fc, plot = False)
            else:
                signal_das_filtered = signal_das

            signal_das_filtered = pra.normalize(signal_das_filtered, 16)

        # draw_spectogram(signal_das, fs, fft_size, fft_hop, analysis_window)
        
            sf.write(f'./results/3d/beamforming/20220315_{algoritm}_source{i}_frec{fc}.wav', signal_das_filtered.astype(np.int16), Fs)
            # sf.write(f'./results/3d/beamforming/rake_perceptual_source{i}_frec{fc}.wav', signal_das_filtered.astype(np.int16), fs)



    # dSNR = pra.dB(room.direct_snr(room.mic_array.center[:, 0], source=0), power=True)
    # print("The direct SNR for good source is " + str(dSNR))

    # S = librosa.feature.melspectrogram(y=out_Das, sr=Fs,n_fft=fft_size,hop_length=fft_hop, n_mels=128,window=analysis_window) 
    
    # log_S = librosa.amplitude_to_db(S, ref=np.max)
    # plt.figure(figsize=(12, 4))
    # librosa.display.specshow(log_S, sr=Fs, x_axis='time', y_axis='mel')
    # plt.title('mel power spectrogram')
    # plt.colorbar(format='%+02.0f dB')
    # plt.tight_layout()
    # plt.savefig(path + "/output_samples/spectrograms_Das_45ms.png", dpi=150)
    # plt.show()


    # # https://github.com/LCAV/pyroomacoustics/issues/137 에 따르면 
    # # 3개의 소스에 대해서 한 소스로만 빔포밍을 하는 것을 불가능하고
    # # rake_mvdr_filters는 두개의 소스에 대한 빔포밍 함수임을 알 수 있다.
    # # 이에 따라 signal1과 signal2를 비슷한 위치 (1,4,1.5) (1,3,1.5)에 두어 signal1으로 빔포밍하여도 signal2도 빔포밍 받은 효과를 줄 수 있다.
