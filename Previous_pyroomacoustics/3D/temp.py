from re import X
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from pyroomacoustics.transform import stft
import pyroomacoustics as pra


def plot_mp3_matplot(filename):

    # sr is for 'sampling rate'
    # Feel free to adjust it
    x, sr = librosa.load(filename, sr=44100)
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)

def convert_audio_to_spectogram(filename, title):

    # sr == sampling rate 
    x, sr = librosa.load(filename, sr=44100)
    
    # stft is short time fourier transform
    X = librosa.stft(x)
    
    # convert the slices to amplitude
    Xdb = librosa.amplitude_to_db(abs(X))
    
    # ... and plot, magic!
    plt.figure(figsize=(14, 5))
    plt.title(title)
    librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'hz')
    plt.colorbar()

    return x
    
# same as above, just changed the y_axis from hz to log in the display func    
def convert_audio_to_spectogram_log(filename, title):
    x, sr = librosa.load(filename, sr=44100)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    plt.title(title)
    librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'log')
    plt.colorbar()

    return x


def rms(x):
    result = np.sqrt(np.mean(x**2))
    return result


def analyze_signals(sig_list, title_list):
    # sig_max = []
    # sig_min = []
    # rms = []

    for path, title in zip(sig_list, title_list):
        x = convert_audio_to_spectogram_log(path, str(title))
        sig_max = np.max(x)
        sig_min = np.min(x)
        rms = np.sqrt(np.mean(x**2))
        energy = np.mean(x ** 2)
        print(f'distance = {title:10f}, sig_max = {sig_max:10f}, sig_min = {sig_min:10f}, rms = {rms}, energy = {energy:.3e}')
    plt.show()

if __name__ == "__main__":

    # wav_path_list = [
    #     "./dataset/original/dji_P4Pro.wav",
    #     # "./dataset/original/Machine_sign.wav",
    #     # './results/2d/beamforming/20220315_doa_source0_frec0.wav',
    #     # './results/2d/beamforming/20220315_doa_source1_frec0.wav'
    #             ]

    # for path in wav_path_list:
    #     convert_audio_to_spectogram(path)
    # # #     sample_rate, samples = wavfile.read(path)
    # # #     print(sample_rate)
    # # #     if len(samples.shape) == 2:
    # # #         samples = samples[...,0]
    # # #     frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    # # #     print(times)
    # # #     plt.figure()
    # # #     plt.pcolormesh(times, frequencies, spectrogram)
    # # #     plt.imshow(spectrogram)
    # # #     plt.title(path)
    # # #     plt.ylabel('Frequency [Hz]')
    # # #     plt.xlabel('Time [sec]')
    # plt.show()

    # fs1, signal1 = wavfile.read("./results/3d/beamforming/drone/all_mix_2m.wav")
    # fs2, signal2 = wavfile.read("./results/3d/beamforming/drone/all_mix_4m.wav")

    # print(rms(signal1))
    # print(rms(signal2))
    # print(rms(signal1) / rms(signal2))


    # fs1, signal1 = wavfile.read("./results/3d/beamforming/drone/drone_test_source_0_2m.wav")
    # fs2, signal2 = wavfile.read("./results/3d/beamforming/drone/drone_test_source_0_4m.wav")

    # print(rms(signal1))
    # print(rms(signal2))
    # print(rms(signal1) / rms(signal2))
    title_list = [10, 50, 100, 500, 1000, 2000, 4000]
    filename_list = [
        # './dataset/original/dji_marvicpro.wav',
        './results/2d/beamforming/drone/drone_test_source_0_10m.wav',
        './results/2d/beamforming/drone/drone_test_source_0_50m.wav',
        './results/2d/beamforming/drone/drone_test_source_0_100m.wav',
        './results/2d/beamforming/drone/drone_test_source_0_500m.wav',
        './results/2d/beamforming/drone/drone_test_source_0_1000m.wav',
        './results/2d/beamforming/drone/drone_test_source_0_2000m.wav',
        './results/2d/beamforming/drone/drone_test_source_0_4000m.wav',
        ]
    
    # filename_list = [
    #     './results/2d/beamforming/drone/all_mix_10m.wav',
    #     './results/2d/beamforming/drone/all_mix_50m.wav',
    #     './results/2d/beamforming/drone/all_mix_100m.wav',
    #     './results/2d/beamforming/drone/all_mix_500m.wav',
    #     './results/2d/beamforming/drone/all_mix_1000m.wav',
    #     './results/2d/beamforming/drone/all_mix_2000m.wav',
    #     './results/2d/beamforming/drone/all_mix_4000m.wav',
    #     ]

    analyze_signals(filename_list, title_list)

    # for path, title in zip(filename_list, title_list):
    #     convert_audio_to_spectogram(path, str(title))
    
    plt.show()

    ##########################################################################################################################################
    # figsize = (15, 7)  # figure size
    # fft_size = 512  # fft size for analysis
    # fft_hop = 8  # hop between analysis frame
    # fft_zp = 512  # zero padding
    # analysis_window = pra.hann(fft_size)
    # t_cut = 0.83  # length in [s] to remove at end of signal (no sound)
    # Fs = 8000

    

    # cmap = "afmhot"
    # interpolation = "none"

    # # Some simulation parameters

    # rate1, signal1 = wavfile.read('./results/2d/beamforming/drone/drone_test_source_0_10m.wav')
    # rate1, signal2 = wavfile.read('./results/2d/beamforming/drone/drone_test_source_0_50m.wav')
    # rate1, signal3 = wavfile.read('./results/2d/beamforming/drone/drone_test_source_0_100m.wav')
    # rate1, signal4 = wavfile.read('./results/2d/beamforming/drone/drone_test_source_0_500m.wav')
    # rate1, signal5 = wavfile.read('./results/2d/beamforming/drone/drone_test_source_0_1000m.wav')
    # rate1, signal6 = wavfile.read('./results/2d/beamforming/drone/drone_test_source_0_2000m.wav')
    # rate1, signal7 = wavfile.read('./results/2d/beamforming/drone/drone_test_source_0_4000m.wav')

    # F0 = stft.analysis(signal1, fft_size, fft_hop, win=analysis_window, zp_back=fft_zp)
    # F1 = stft.analysis(signal2, fft_size, fft_hop, win=analysis_window, zp_back=fft_zp)
    # F2 = stft.analysis(signal3, fft_size, fft_hop, win=analysis_window, zp_back=fft_zp)
    # F3 = stft.analysis(signal4, fft_size, fft_hop, win=analysis_window, zp_back=fft_zp)
    # F4 = stft.analysis(signal5, fft_size, fft_hop, win=analysis_window, zp_back=fft_zp)
    # F5 = stft.analysis(signal6, fft_size, fft_hop, win=analysis_window, zp_back=fft_zp)
    # F6 = stft.analysis(signal7, fft_size, fft_hop, win=analysis_window, zp_back=fft_zp)

    # p_min = 7
    # p_max = 100
    # all_vals = np.concatenate(
    #     (
    #         pra.dB(F0 + pra.eps),
    #         pra.dB(F1 + pra.eps),
    #         pra.dB(F2 + pra.eps),
    #         pra.dB(F3 + pra.eps),
    #         pra.dB(F4 + pra.eps),
    #         pra.dB(F5 + pra.eps),
    #         pra.dB(F6 + pra.eps),
    #     )
    # ).flatten()
    # vmin, vmax = np.percentile(all_vals, [p_min, p_max])

    # fig, ax = plt.subplots(figsize=figsize, nrows=2, ncols=3)

    # ax = plt.subplot(2, 3, 1)
    # plot_spectrogram(F0, "10m")

    # ax = plt.subplot(2, 3, 4)
    # plot_spectrogram(F1, "50m")

    # ax = plt.subplot(2, 3, 2)
    # plot_spectrogram(F2, "100m")

    # ax = plt.subplot(2, 3, 5)
    # plot_spectrogram(F3, "500m")

    # ax = plt.subplot(2, 3, 3)
    # plot_spectrogram(F4, "1000m")

    # ax = plt.subplot(2, 3, 6)
    # plot_spectrogram(F5, "2000m")

    # plt.show()
    