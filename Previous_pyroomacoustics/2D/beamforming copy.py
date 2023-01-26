import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
from pyroomacoustics.transform import stft
from scipy.io import wavfile
import soundfile as sf
import samplerate
from utils import preprocess_signal, cut_signals, draw_spectogram, DoughertyLogSpiral

# specify signal and noise source
fs1, signal1 = wavfile.read("./dataset/original/Machine.wav")
fs2, signal2 = wavfile.read("./dataset/original/Machine.wav")
fs3, signal3 = wavfile.read("./dataset/original/Machine.wav")
fs4, signal4 = wavfile.read("./dataset/original/Machine_sign.wav")

signal1 = signal1[:fs1*5]
signal2 = signal2[:fs2*5]
signal3 = signal3[:fs3*5]
signal4 = signal4[:fs4*5]


fs = fs4
signal1 = preprocess_signal(fs1,fs,signal1)
signal2 = preprocess_signal(fs2,fs,signal2)
signal3 = preprocess_signal(fs3,fs,signal3)
signal4 = preprocess_signal(fs4,fs,signal4)

signals = cut_signals([signal1, signal2, signal3, signal4])

Lg_t = 0.100      
Lg = np.ceil(Lg_t*fs)   

sigma2_n = 5e-7

room_bf = pra.ShoeBox([5,10], fs=fs, max_order=0, sigma2_awgn=sigma2_n)  

source1_loc = np.array([2.5, 2])
source2_loc = np.array([2.5, 4])
source3_loc = np.array([2.5, 6])
source4_loc = np.array([2.5, 8])

room_bf.add_source(source1_loc, delay=0., signal=signals[0])
# room_bf.add_source(source2_loc, delay=0., signal=signals[1])
# room_bf.add_source(source3_loc, delay=0., signal=signals[2])
room_bf.add_source(source4_loc, delay=0., signal=signals[3])

# Create geometry equivalent to Amazon Echo
center = [1, 5]; radius = 50e-3
fft_len = 1024 
# mics = pra.circular_2D_array(center=center, M=30, phi0=0, radius=radius)
mics = DoughertyLogSpiral(center, rmax = 0.15, rmin = 0.025, v = 87 *np.pi/180, n_mics = 15, direction = '2d')
# echo = np.concatenate((echo, np.array(center, ndmin=2).T), axis=1)
# mics = np.concatenate((mics, np.array(center, ndmin=2).T), axis=1)
mics = pra.Beamformer(mics, room_bf.fs, N=fft_len, Lg=Lg)

room_bf.add_microphone_array(mics)


room_bf.compute_rir()
room_bf.simulate()
room_bf.plot()
plt.show()


sf.write('./results/2d/beamforming/20220315_all_mix.wav', room_bf.mic_array.signals[-1,:].astype(np.int16),  fs)
# Compute DAS weights


plot = True

for i in range(2):
    # if i != 3:
    #     # room_bf.mic_array.rake_mvdr_filters(room_bf.sources[i][:2], room_bf.sources[i+1][:2], sigma2_n * np.eye(mics.Lg * mics.M), delay=0.0, epsilon=0.005)
    #     room_bf.mic_array.rake_delay_and_sum_weights(room_bf.sources[i][:2], room_bf.sources[i+1][:2])
    # else:
    #     # room_bf.mic_array.rake_mvdr_filters(room_bf.sources[3][:2], room_bf.sources[2][:2], sigma2_n * np.eye(mics.Lg * mics.M), delay=0.0, epsilon=0.005)
    #     room_bf.mic_array.rake_delay_and_sum_weights(room_bf.sources[3][:2], room_bf.sources[2][:2])

    room_bf.mic_array.rake_mvdr_filters(room_bf.sources[i][:2], interferer=room_bf.sources[1-i][:2], R_n= sigma2_n * np.eye(mics.Lg * mics.M), delay=0)
    if plot:
        fig, ax = room_bf.plot(freq=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], img_order=0)
        ax.legend(['1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000'])
        # fig.set_size_inches(20, 8)
        # ax.set_xlim([-3,8])
        # ax.set_ylim([-3,8])
        plt.show()
        
    
    signal_das = room_bf.mic_array.process(FD=False)
    cut_fcs = [0,2000,4000,5000,6000, 7000] 
    cut_fcs = [0] 

    for fc in cut_fcs:
        if fc != 0:
            signal_das_filtered = pra.highpass(signal_das, fs, fc = fc, plot = False)
        else:
            signal_das_filtered = signal_das

        signal_das_filtered = pra.normalize(signal_das_filtered, 16)

        # draw_spectogram(signal_das, fs, fft_size, fft_hop, analysis_window)
        # plt.show()
        sf.write(f'./results/2d/beamforming/20220315_doa_source{i}_frec{fc}.wav', signal_das_filtered.astype(np.int16), fs)
        # sf.write(f'./results/2d/beamforming/rake_perceptual_source{i}_frec{fc}.wav', signal_das_filtered.astype(np.int16), fs)



    # t60 = pra.experimental.measure_rt60(room_bf.rir[0][0], fs=room_bf.fs, plot=True)
