import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve,resample
import pyroomacoustics as pra

corners = np.array([[0,0], [0,3], [5,3], [5,1], [3,1], [3,0]]).T  # [x,y]
room = pra.Room.from_corners(corners)

fs, signal = wavfile.read("./input/male_voice.wav")

room = pra.Room.from_corners(corners, fs=fs, max_order=10, materials=pra.Material(0.2, 0.15), ray_tracing=True, air_absorption=True)
room.extrude(2., materials=pra.Material(0.2, 0.15))

room.set_ray_tracing(receiver_radius=0.5, n_rays=10000, energy_thres=1e-5)

room.add_source([1., 1., 0.5], signal=signal)

R = np.array([[3.5, 3.6], [2., 2.], [0.5,  0.5]])
room.add_microphone(R)

room.image_source_model()

fig, ax = room.plot(img_order=3)
fig.set_size_inches(18.5, 10.5)

room.plot_rir()
fig = plt.gcf()
fig.set_size_inches(20, 10)


room.simulate()
print(room.mic_array.signals.shape)
sf.write(r'C:\Users\bob04\pyroom\rir_output\org.wav', signal, fs)
sf.write(r'C:\Users\bob04\pyroom\rir_output\rir_signal.wav', room.mic_array.signals[0,:], fs)


#잔향 효과로 인해 큰 주파수 대역 소리들로 합쳐지는 듯 함


