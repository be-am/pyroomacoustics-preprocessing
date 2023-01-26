import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
from scipy.io import wavfile
import soundfile as sf

# specify signal and noise source
fs, signal = wavfile.read("input\girl.wav")
fs2, noise = wavfile.read("input\cafe.wav")  # may spit out a warning when reading but it's alright!
signal = np.squeeze(signal[:,0])
noise = np.squeeze(noise[:,0])

sig_length = np.min([signal.shape[0], noise.shape[0]])
noise = noise[:sig_length] 
signal = signal[:sig_length]


Lg_t = 0.100                # filter size in seconds
Lg = np.ceil(Lg_t*fs)       # in samples

# Create 4x6 shoebox room with source and interferer and simulate
room_bf = pra.ShoeBox([4,6], fs=fs, max_order=0)                #max_order 가 t60의 시간임 작을수록 잔향 효과가 빨리 끝남
source = np.array([1, 4.5])
interferer = np.array([3.5, 3.])
room_bf.add_source(source, delay=0., signal=signal)
room_bf.add_source(interferer, delay=0., signal=noise)

# Create geometry equivalent to Amazon Echo
center = [2, 1.5]; radius = 37.5e-3
fft_len = 512
echo = pra.circular_2D_array(center=center, M=6, phi0=0, radius=radius)
echo = np.concatenate((echo, np.array(center, ndmin=2).T), axis=1)
mics = pra.Beamformer(echo, room_bf.fs, N=fft_len, Lg=Lg)
room_bf.add_microphone_array(mics)

# Compute DAS weights
room_bf.mic_array.rake_delay_and_sum_weights(room_bf.sources[0][:1])

# plot the room and resulting beamformer before simulation
fig, ax = room_bf.plot(freq=[500, 1000, 2000, 4000], img_order=0)
ax.legend(['500', '1000', '2000', '4000'])
fig.set_size_inches(20, 8)
ax.set_xlim([-3,8])
ax.set_ylim([-3,8])

room_bf.compute_rir()
room_bf.simulate()
sf.write(r'C:\Users\bob04\pyroom\das_beamforming_output\all_mix.wav', room_bf.mic_array.signals[-1,:],  fs)

signal_das = room_bf.mic_array.process(FD=False)
sf.write(r'C:\Users\bob04\pyroom\das_beamforming_output\das_beamforming.wav', signal_das, fs)

t60 = pra.experimental.measure_rt60(room_bf.rir[0][0], fs=room_bf.fs, plot=True)
