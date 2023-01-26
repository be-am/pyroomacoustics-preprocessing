from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra
import soundfile as sf
import samplerate
from scipy import signal
from scipy.signal import butter, lfilter

# def FFT(Vb_sig, Ts):
#     Fs = 1/Ts
#     n = len(Vb_sig)
#     k = np.arange(n)
#     T = n/Fs
#     freq = k/T              
#     freq = freq[range(int(n/2))]        
#     Y = np.fft.fft(Vb_sig)/n
#     Y = Y[range(int(n/2))]
#     return Y

# def butter_bandpass(lowcut, highcut, fs, order):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a
# def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#     b,a = butter_bandpass(lowcut, highcut, fs, order)
#     y = lfilter(b,a,data)
#     return y

# fs, noise = wavfile.read("./input/male_voice.wav")
# fs, signal = wavfile.read("./input/female_voice.wav")  # may spit out a warning when reading but it's alright!

sr_s, signal = wavfile.read("./input/female_voice.wav")  # may spit out a warning when reading but it's alright!
sr_n, noise = wavfile.read("./input/male_voice.wav")
signal = np.squeeze(signal[:,0])
noise = np.squeeze(noise[:,0])

new_rate = 8000
fs_ratio_n = new_rate / float(sr_n)
fs_ratio_s = new_rate / float(sr_s)
signal = samplerate.resample(signal, fs_ratio_s, "sinc_best")
noise = samplerate.resample(noise, fs_ratio_n, "sinc_best")


sig_length = np.min([signal.shape[0], noise.shape[0]])
noise = noise[:sig_length] 
signal = signal[:sig_length]

# fft_sig = FFT(signal, 5)
# plt.plot(abs(fft_sig))
# plt.show()

# filter_ = signal.firwin(101, cutoff=5000, fs = fs, pass_zero='lowpass')
# Vb_sig_sum = lfilter(filter_, [1,0], signal)
# sf.write(r'C:\Users\bob04\pyroom\mv-beam\output_samples\lp_sig.wav',Vb_sig_sum, fs)

# fft_sig = FFT(Vb_sig_sum, 5)
# plt.plot(abs(fft_sig))
# plt.show()

# Create 4x6 shoebox room with source and interferer and simulate
room_mv_bf = pra.ShoeBox([4,6], fs=new_rate, max_order=0)
source = np.array([1, 4.5])
interferer = np.array([3.5, 3.])
room_mv_bf.add_source(source, delay=0., signal=signal)
room_mv_bf.add_source(interferer, delay=0., signal=noise)

center = [2, 1.5]; radius = 37.5e-3
fft_len = 512
echo = pra.circular_2D_array(center=center, M=6, phi0=0, radius=radius)
echo = np.concatenate((echo, np.array(center, ndmin=2).T), axis=1)
mics = pra.Beamformer(echo, room_mv_bf.fs, N=fft_len)
room_mv_bf.add_microphone_array(mics)

mic_noise = 30
R_n = 10**((mic_noise-94)/20)*np.eye(fft_len*room_mv_bf.mic_array.M)
room_mv_bf.mic_array.rake_mvdr_filters(room_mv_bf.sources[0][:1], interferer = room_mv_bf.sources[1][:1], R_n = R_n)

fig, ax = room_mv_bf.plot(freq = [500, 1000, 2000 , 4000], img_order=0)
ax.legend(['500', '1000', '2000', '4000'])
fig.set_size_inches(20, 8)
ax.set_xlim([-3,8])
ax.set_ylim([-3,8])
room_mv_bf.compute_rir()
room_mv_bf.simulate()

# sf.write(r'C:\Users\bob04\pyroom\mv-beam\output_samples\all_mix.wav', room_mv_bf.mic_array.signals[-1,:],  fs)
wavfile.write("./mv-beam/output_samples/all_mix.wav", new_rate, room_mv_bf.mic_array.signals[-1,:].astype(np.int16))

#beamforming process
sig_mv = room_mv_bf.mic_array.process(FD=False)
out_mv = pra.normalize(pra.highpass(sig_mv, 700))
# sf.write(r'C:\Users\bob04\pyroom\mv-beam\output_samples\mv_beamforming.wav', sig_mv, fs)
wavfile.write("./mv-beam/output_samples/mv_beamforming.wav", new_rate,sig_mv.astype(np.int16))

