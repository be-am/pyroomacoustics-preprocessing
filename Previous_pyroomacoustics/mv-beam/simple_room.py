from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
from scipy.io import wavfile
import soundfile as sf
from scipy.signal import resample
import samplerate



sr_s, signal = wavfile.read("./input/female_google.wav")  # may spit out a warning when reading but it's alright!


# new_rate = 8000

new_rate = 8000
fs_ratio = new_rate / float(sr_s)

signal = samplerate.resample(signal, fs_ratio, "sinc_best")
sf.write('./signal.wav', signal,  new_rate)

input()
# number_of_samples_n = round(len(noise) * float(new_rate) / sr_n)
# number_of_samples_s = round(len(signal) * float(new_rate) / sr_s)
# noise = resample(noise, number_of_samples_n)
# signal = resample(signal, number_of_samples_s)
# signal = np.squeeze(signal[:,0])
# noise = np.squeeze(noise[:,0])



# Create 4x6 shoebox room with source and interferer and simulate
# room_mv_bf = pra.ShoeBox([4,6], fs=new_rate, max_order=0)
# source = np.array([1, 4.5])
# room_mv_bf.add_source(source, delay=0., signal=signal)

corners = np.array([[0,0], [0,3], [5,3], [5,1], [3,1], [3,0]]).T
room = pra.Room.from_corners(corners, fs=new_rate, max_order=3, materials=pra.Material(0.2, 0.15), ray_tracing=True, air_absorption=True)
room.extrude(2., materials=pra.Material(0.2, 0.15))
room.set_ray_tracing(receiver_radius=0.5, n_rays=10000, energy_thres=1e-5)

room.add_source([1., 1., 0.5], signal=signal)

R = np.array([[3.5, 3.6], [2., 2.], [0.5,  0.5]])  # [[x], [y], [z]]
room.add_microphone(R)

room.image_source_model()
fig, ax = room.plot(img_order=3)
fig.set_size_inches(18.5, 10.5)
# plt.show()

room.simulate()

sf.write('./signal.wav', signal,  new_rate)
sf.write('./all_mix.wav', room.mic_array.signals[0,:],  new_rate)



