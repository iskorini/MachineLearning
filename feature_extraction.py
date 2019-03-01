import numpy as np
import glob
import speechpy as sphp
import scipy.io.wavfile as wav
from sys import exit
from pathlib import Path
import pandas as pd

def extract_feature_from_wav(path):
    (rate, sig) = wav.read(path)
    mfcc_feat = sphp.feature.mfcc(sig, sampling_frequency=16000, frame_length=0.025,
                              frame_stride=0.01, num_filters=26, num_cepstral=12)
    log_energy = sphp.feature.lmfe(sig, sampling_frequency=16000, frame_length=0.025,
                               frame_stride=0.01, num_filters=26)
    feature = np.append(mfcc_feat, np.sum(log_energy, axis=1).
                        reshape(log_energy.shape[0], 1), axis=1)
    return np.reshape(sphp.feature.extract_derivative_feature(feature)[:, :, 0:2], (log_energy.shape[0], 26))


for file_name in glob.iglob('./timit/**/**/**/*CONVERTED.wav'):
    feature = extract_feature_from_wav(file_name)
    feature_name = file_name.replace('timit', 'preprocessed_dataset')
    feature_name = feature_name.replace('wav', 'csv')
    feature_name = feature_name.replace('CONVERTED', '')
    Path(feature_name[:''.join(feature_name).rindex('/')]).mkdir(parents=True, exist_ok=True)
    print(file_name, " -> ", feature_name)
    pd.DataFrame(feature).to_csv(feature_name)

exit(0)