import numpy as np
import glob
import speechpy as sphp
import scipy.io.wavfile as wav
from sys import exit
from pathlib import Path
import pandas as pd

NUM_FILTERS = 40
FRAME_STRIDE = 0.01 #seconds
FRAME_LENGTH = 0.025 #seconds
NUM_CEPSTRAL = 12


def extract_feature_from_wav(path):
    (rate, sig) = wav.read(path)
    log_energy = sphp.feature.mfcc(sig, sampling_frequency=16000, frame_length=FRAME_LENGTH,
                               frame_stride=FRAME_STRIDE, num_filters=NUM_FILTERS)
    return sphp.feature.extract_derivative_feature(log_energy)


def create_file_of_feature(path):
    for file_name in glob.iglob(path):
        feature_name = file_name.replace('timit', 'preprocessed_dataset_MFCC')
        feature_name = feature_name.replace('.wav', '.csv')
        feature_name = feature_name.replace('CONVERTED', '')
        Path(feature_name[:''.join(feature_name).rindex('/')]).mkdir(parents=True, exist_ok=True)
        print(file_name, " -> ", feature_name)
        extracted_feature = extract_feature_from_wav(file_name)
        feature = pd.DataFrame(np.concatenate((extracted_feature[:, :, 0],
                                            extracted_feature[:, :, 1],
                                            extracted_feature[:, :, 2]), axis=1))
        feature.rename(index=str, columns={'Unnamed: 0': 'frame'})
        feature['start_frame'] = np.arange(len(feature))
        feature['start_frame'] = feature['start_frame'] * FRAME_STRIDE * 1000
        feature['end_frame'] = feature['start_frame'] + FRAME_LENGTH * 1000
        phoneme_name = file_name.replace('CONVERTED', '')
        phoneme_name = phoneme_name.replace('wav', 'phn')
        file = open(phoneme_name, 'r')
        phoneme = file.read()
        file.close()
        phoneme = np.array([[e for e in line.split(' ')] for line in phoneme.splitlines()])
        phoneme[:, 0:2] = (phoneme[:, 0:2].astype(int)) // 16  # convert to ms (sampling rate 16KHz)
        feature['phoneme'] = 'h#'
        for row in phoneme:
            feature.loc[feature['start_frame'] > int(row[0]), 'phoneme'] = row[2]
        feature.to_csv(feature_name)

if __name__ == '__main__':
    create_file_of_feature('./timit/**/**/**/**CONVERTED.wav')
    exit(0)