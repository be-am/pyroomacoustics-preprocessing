import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve,resample
import pyroomacoustics as pra

fs, my_signal = wavfile.read("./input/male_voice.wav")
room = pra.ShoeBox([10, 5, 3.2], fs=16000, absorption=0.25, max_order=10)

# add one source at a time, with source signal
room.add_source([2.5, 1.7, 1.69], signal=my_signal)
# add microphone array, R.shape == (3, n_mics)
R = np.array([[5.71, 2.31, 1.4], [5.72, 2.32, 1.4]]).T
room.add_microphone_array(pra.MicrophoneArray(R, fs=room.fs))

room.simulate()
output_signal = room.mic_array.signals       # (n_mics, n_samples)
room.plot(img_order=2)      # show room
room.plot_rir()     # show RIR


sf.write(r'C:\Users\bob04\pyroom\rir_acc_output\rir_acc.wav',output_signal[-1,:],fs)

t60 = pra.experimental.measure_rt60(room.rir[0][0], fs=room.fs, plot=True)