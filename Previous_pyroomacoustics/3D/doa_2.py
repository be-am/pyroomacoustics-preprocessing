# -*- coding: utf-8 -*-
"""
DOA Algorithms
==============
This example demonstrates how to use the DOA object to perform direction of arrival
finding in 2D using one of several algorithms
- MUSIC [1]_
- SRP-PHAT [2]_
- CSSM [3]_
- WAVES [4]_
- TOPS [5]_
- FRIDA [6]_
.. [1] R. Schmidt, *Multiple emitter location and signal parameter estimation*, 
    IEEE Trans. Antennas Propag., Vol. 34, Num. 3, pp 276--280, 1986
.. [2] J. H. DiBiase, J H, *A high-accuracy, low-latency technique for talker localization 
    in reverberant environments using microphone arrays*, PHD Thesis, Brown University, 2000
.. [3] H. Wang, M. Kaveh, *Coherent signal-subspace processing for the detection and 
    estimation of angles of arrival of multiple wide-band sources*, IEEE Trans. Acoust., 
    Speech, Signal Process., Vol. 33, Num. 4, pp 823--831, 1985
.. [4] E. D. di Claudio, R. Parisi, *WAVES: Weighted average of signal subspaces for 
    robust wideband direction finding*, IEEE Trans. Signal Process., Vol. 49, Num. 10, 
    2179--2191, 2001
.. [5] Y. Yeo-Sun, L. M. Kaplan, J. H. McClellan, *TOPS: New DOA estimator for wideband 
    signals*, IEEE Trans. Signal Process., Vol. 54, Num 6., pp 1977--1989, 2006
.. [6] H. Pan, R. Scheibler, E. Bezzam, I. Dokmanić, and M. Vetterli, *FRIDA:
    FRI-based DOA estimation for arbitrary array layouts*, Proc. ICASSP,
    pp 3186-3190, 2017
In this example, we generate some random signal for a source in the far field
and then simulate propagation using a fractional delay filter bank
corresponding to the relative microphone delays.
Then we perform DOA estimation and compare the errors for different algorithms
"""

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

import pyroomacoustics as pra
from pyroomacoustics.doa import circ_dist
from scipy.io import wavfile
import os

def preprocess_data(signal, Fs):
    np.array(signal, dtype=float)
    signal = pra.highpass(signal, Fs)
    signal = pra.normalize(signal)

    if np.shape(signal)[1]:
        signal = np.squeeze(signal[:,0])
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
    ######
    # We define a meaningful distance measure on the circle

    # Location of original source
    azimuth = 61.0 / 180.0 * np.pi  # 60 degrees
    distance = 3.0  # 3 meters

    #######################
    # algorithms parameters
    SNR = 0.0  # signal-to-noise ratio
    c = 343.0  # speed of sound
    fs = 8000  # sampling frequency
    nfft = 1024  # FFT size
    freq_bins = np.arange(5, 60)  # FFT bins to use for estimation

    # compute the noise variance
    sigma2 = 10 ** (-SNR / 10) / (4.0 * np.pi * distance) ** 2

    # Create an anechoic room
    room_dim = [10.0, 5.0,5.0]
    aroom = pra.ShoeBox(room_dim, fs=fs, max_order=0, sigma2_awgn=sigma2)

    # add the source
    source_location = [1,3,1.5]
    path = os.path.dirname(__file__) 

    rate, source_signal = wavfile.read(path + "/input_samples/female_voice.wav")  # may spit out a warning when reading but it's alright!
    source_signal = preprocess_data(source_signal,fs)
    aroom.add_source(source_location, signal=source_signal)

    # We use a circular array with radius 15 cm # and 4 microphones
    mic_center = np.array([8, 3, 1])
    mic_radius = 0.05
    R = circular_3d_coords(mic_center, mic_radius, 4, 'vertical')
    mics = pra.Beamformer(R, fs, 1024, Lg=np.ceil(0.1 * fs))

    aroom.add_microphone_array(mics)

    # run the simulation
    aroom.simulate()

    ################################
    # Compute the STFT frames needed
    X = np.array(
        [
            pra.transform.stft.analysis(signal, nfft, nfft // 2).T
            for signal in aroom.mic_array.signals
        ]   
    )
    print(np.shape(X))

    ##############################################
    # Now we can test all the algorithms available
    doa = pra.doa.MUSIC(R,fs,nfft)

    # this call here perform localization on the frames in X
    doa.locate_sources(X, freq_bins=freq_bins)

    doa.polar_plt_dirac()
    plt.title('MUSIC')

    # doa.azimuth_recon contains the reconstructed location of the source
    print("  Recovered azimuth:", doa.azimuth_recon / np.pi * 180.0, "degrees")
    print("  Error:", circ_dist(azimuth, doa.azimuth_recon) / np.pi * 180.0, "degrees")

    # for algo_name in algo_names:
    #     # Construct the new DOA object
    #     # the max_four parameter is necessary for FRIDA only
    #     doa = pra.doa.algorithms[algo_name](R, fs, nfft, c=c, max_four=4)

    #     # this call here perform localization on the frames in X
    #     doa.locate_sources(X, freq_bins=freq_bins)

    #     doa.polar_plt_dirac()
    #     plt.title(algo_name)

    #     # doa.azimuth_recon contains the reconstructed location of the source
    #     print(algo_name)
    #     print("  Recovered azimuth:", doa.azimuth_recon / np.pi * 180.0, "degrees")
    #     print("  Error:", circ_dist(azimuth, doa.azimuth_recon) / np.pi * 180.0, "degrees")

    plt.show()