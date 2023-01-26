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
# fs1, signal1 = wavfile.read("./dataset/smi_data/cafe_music_5min.wav")
fs2, signal2 = wavfile.read("./dataset/smi_data/woman_5min.wav")
fs3, signal3 = wavfile.read("./dataset/smi_data/man1_5min.wav")
# fs3, signal3 = wavfile.read("./dataset/smi_data/man2_5min.wav")

# fs4, noise = wavfile.read("./dataset/smi_data/cafe_noise_5min.wav")

fs = fs2
# signal1 = preprocess_signal(fs1,fs,signal1)
signal2 = preprocess_signal(fs2,fs,signal2)
signal3 = preprocess_signal(fs3,fs,signal3)
# noise = preprocess_signal(fs4,fs,noise)

# signals = cut_signals([signal1, signal2, signal3, noise])
signals = cut_signals([signal2, signal3])

Lg_t = 0.100      
Lg = np.ceil(Lg_t*fs)   

# Spectrogram figure properties
figsize = (15, 7)  # figure size
fft_size = 512  # fft size for analysis
fft_hop = 8  # hop between analysis frame
fft_zp = 512  # zero padding
analysis_window = pra.hann(fft_size)
t_cut = 0.83  # length in [s] to remove at end of signal (no sound)

# draw_spectogram(signal3, fs, fft_size, fft_hop, analysis_window)
# plt.show()
sigma2_n = 5e-7

room_bf = pra.ShoeBox([7.5, 5.7], fs=fs, max_order=0, sigma2_awgn=sigma2_n)  
source1 = np.array([5.25, 1.12])
source2 = np.array([5.25, 2.85])
source3 = np.array([5.25, 4.58])
interferer = np.array([3.5, 5])

room_bf.add_source(source1, delay=0., signal=signals[0])
room_bf.add_source(source2, delay=0., signal=signals[1])
# room_bf.add_source(source3, delay=0., signal=signals[2])
# room_bf.add_source(interferer, delay=0., signal=signals[3])

# Create geometry equivalent to Amazon Echo
center = [2.25, 2.85]; radius = 50e-3
fft_len = 1024
mics = pra.circular_2D_array(center=center, M=30, phi0=0, radius=radius)
# mics = DoughertyLogSpiral(center, rmax = 0.15, rmin = 0.025, v = 87 *np.pi/180, n_mics = 112, direction = '2d')
# echo = np.concatenate((echo, np.array(center, ndmin=2).T), axis=1)

mics = pra.Beamformer(mics, room_bf.fs)#, N=fft_len, Lg=Lg)
room_bf.add_microphone_array(mics)

room_bf.compute_rir()
room_bf.simulate()
# room_bf.plot()
# plt.show()


sf.write('./results/2d/beamforming/all_mix.wav', room_bf.mic_array.signals[-1,:].astype(np.int16),  fs)
# Compute DAS weights


plot = False

for i in range(0,2):
    # room_bf.mic_array.rake_delay_and_sum_weights(room_bf.sources[i][:2])
    room_bf.mic_array.rake_perceptual_filters(room_bf.sources[i][:2], None, sigma2_n * np.eye(mics.Lg * mics.M), delay=0)


    # mics.rake_perceptual_filters(
    #     good_sources, bad_sources1, sigma2_n * np.eye(mics.Lg * mics.M), delay=0
    # )

    if plot:
        fig, ax = room_bf.plot(freq=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], img_order=0)
        ax.legend(['1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000'])
        fig.set_size_inches(20, 8)
        ax.set_xlim([-3,8])
        ax.set_ylim([-3,8])
        
    
    signal_das = room_bf.mic_array.process(FD=False)
    # cut_fcs = [0,2000,4000,5000,6000, 7000] 
    cut_fcs = [0] 
    for fc in cut_fcs:
        if fc != 0:
            signal_das_filtered = pra.highpass(signal_das, fs, fc = fc, plot = False)
        else:
            signal_das_filtered = signal_das

        signal_das_filtered = pra.normalize(signal_das_filtered, 16)

        # draw_spectogram(signal_das, fs, fft_size, fft_hop, analysis_window)
        # plt.show()
        sf.write(f'./results/2d/beamforming/doa_source{i}_frec{fc}.wav', signal_das_filtered.astype(np.int16), fs)
        # sf.write(f'./results/2d/beamforming/rake_perceptual_source{i}_frec{fc}.wav', signal_das_filtered.astype(np.int16), fs)



    # t60 = pra.experimental.measure_rt60(room_bf.rir[0][0], fs=room_bf.fs, plot=True)
