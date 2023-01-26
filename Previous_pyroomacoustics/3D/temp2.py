import numpy as np
import pyroomacoustics as pra

if __name__ == "__main__":

    fs = 16000
    fft_len = 1024
    Lg_t = 0.100
    sigma2_n = 5e-7
    Lg = np.ceil(Lg_t*fs)
    n = fs * 1
    x = np.random.randn(n)

    source = np.array([1.0, 1.0])
    centers = [
            [1.0, 2.0],
            [1.0, 11.0],
            [1.0, 101.0],
            [1.0, 1001.0],
        ]

    signals = []
    energys = []

    for center in centers:
        room = pra.ShoeBox([10000, 10000], max_order=0, fs=fs, sigma2_awgn=sigma2_n)
        room.add_source(source, signal=x)

        R = pra.circular_2D_array(center=center, M=12, phi0=0, radius=0.2)
        mics = pra.Beamformer(R, room.fs, N=fft_len, Lg=Lg)
        room.add_microphone_array(mics)
        room.simulate()

        # room.mic_array.rake_perceptual_filters(room.sources[0][:2], None, sigma2_n * np.eye(mics.Lg * mics.M))
        room.mic_array.rake_delay_and_sum_weights(room.sources[0][:2], None, R_n = sigma2_n * np.eye(mics.Lg * mics.M))
        result_signal = room.mic_array.process(FD=False)
        result_signal = pra.normalize(result_signal, 16)

        signals.append(result_signal)
        energys.append(np.mean(result_signal ** 2))


    mic_1m = centers[0]
    energy_1m = energys[0]

    for idx in range(1, len(centers)):
        dist = np.linalg.norm(centers[idx] - source)
        print(f"distance={dist:.3e} : energy decay = {energy_1m / energys[idx]:.3e}")