import numpy as np
import glob
import speechpy as sphp
import scipy.io.wavfile as wav
from sys import exit
from pathlib import Path
import pandas as pd
import librosa
import os

#FOR PYTHON2
NUM_FILTERS = 40
FRAME_STRIDE = 0.01 #seconds
FRAME_LENGTH = 0.025 #seconds

#test = '/home/fs6185896/ML_PROJECT/MachineLearning/timit/test/dr1/faks0/sa1CONVERTED.wav'
#y, sr = librosa.load(test, sr=16000)
#mfcc = librosa.feature.mfcc(y,sr, n_mfcc=40, hop_length=160, n_fft=400)
#energy  = librosa.feature.mfcc(y,sr, n_mfcc=40, hop_length=160, n_fft=400, power=1)
# https://groups.google.com/forum/#!topic/librosa/V4Z1HpTKn8Q
def extract_feature_from_wav(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y,sr, n_mfcc=40, hop_length=int(sr*FRAME_STRIDE), n_fft=int(sr*FRAME_LENGTH))
    mfcc_delta = librosa.feature.delta(mfcc, order = 1)
    mfcc_delta_delta = librosa.feature.delta(mfcc, order = 2)
    energy  = librosa.feature.rmse(y, frame_length=int(sr*FRAME_LENGTH), hop_length=int(sr*FRAME_STRIDE))
    energy_delta = librosa.feature.delta(energy)
    energy_delta_delta = librosa.feature.delta(energy)
    return np.concatenate((mfcc, energy)), np.concatenate((mfcc_delta, energy_delta)), np.concatenate((mfcc_delta_delta, energy_delta_delta))


def create_file_of_feature(path):
    for file_name in glob.iglob(path):
        feature_name = file_name.replace('timit', 'preprocessed_dataset_LIBROSA')
        feature_name = feature_name.replace('.wav', '.csv')
        feature_name = feature_name.replace('CONVERTED', '')
        pth = Path(feature_name[:''.join(feature_name).rindex('/')])
        if not os.path.exists(str(pth)):
            pth.mkdir(parents=True)
        print(file_name, " -> ", feature_name)
        extracted_feature = extract_feature_from_wav(file_name)
        feature = pd.DataFrame(np.concatenate(extracted_feature).transpose())
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