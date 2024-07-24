import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

if "__main__" == __name__:
    filepath = "/media/zhouying/data/zhouy01/MEAD/M003/audio/angry/level_1/001.m4a"
    signal, sr = librosa.load(path=filepath, sr=16000)
    N_FFT = 512
    N_MELS = 80
    N_MFCC = 13

    mel_spec = librosa.feature.melspectrogram(y=signal,
                                              sr=sr,
                                              n_fft=N_FFT,
                                              hop_length=sr // 100,
                                              win_length=sr // 40,
                                              n_mels=N_MELS)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=N_MFCC) # frames = length(signal)//hop_length + 1

    delta_mfcc = librosa.feature.delta(data=mfcc)
    delta2_mfcc = librosa.feature.delta(data=mfcc, order=2)
    mfcc = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)

    librosa.display.specshow(data=mfcc,
                             sr=sr,
                             n_fft=N_FFT,
                             hop_length=sr // 100,
                             win_length=sr // 40,
                             x_axis="s")
    plt.colorbar(format="%d")

    plt.show()

