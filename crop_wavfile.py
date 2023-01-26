

from numpy import sign
from scipy.io import wavfile
import soundfile as sf
import os

def crop_wav_file(signals, save_path, fs, start_time_list, end_time_list):

    for i, signal in enumerate(signals):
        for j, (start_time, end_time) in enumerate(zip(start_time_list, end_time_list)):
            cut_signal = signal[start_time * fs:end_time*fs]

            sf.write(os.path.join(save_path, f'{i}_{j}.wav'), cut_signal, fs)







if __name__ == "__main__":
    save_path = './dataset/original/'
    fs1, signal1 = wavfile.read("./dataset/original/Ultimate DJI drones sound test.wav")
    start_time_list = [67, 73, 79, 92, 98]
    end_time_list = [72, 78, 84, 96, 102]
    signal1 = signal1[...,0]

    crop_wav_file([signal1], save_path, fs1, start_time_list, end_time_list)