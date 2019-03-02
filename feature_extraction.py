import numpy as np
import glob
import speechpy as sphp
import scipy.io.wavfile as wav
from sys import exit
from pathlib import Path
import pandas as pd
import csv

def extract_feature_from_wav(path):
    (rate, sig) = wav.read(path)
    mfcc_feat = sphp.feature.mfcc(sig, sampling_frequency=16000, frame_length=0.025,
                              frame_stride=0.01, num_filters=26, num_cepstral=12)
    log_energy = sphp.feature.lmfe(sig, sampling_frequency=16000, frame_length=0.025,
                               frame_stride=0.01, num_filters=26)
    feature = np.append(mfcc_feat, np.sum(log_energy, axis=1).
                        reshape(log_energy.shape[0], 1), axis=1)
    return np.reshape(sphp.feature.extract_derivative_feature(feature)[:, :, 0:2], (log_energy.shape[0], 26))


def create_file_of_feature(path):
    for file_name in glob.iglob(path):
        feature = extract_feature_from_wav(file_name)
        feature_name = file_name.replace('timit', 'preprocessed_dataset')
        feature_name = feature_name.replace('wav', 'csv')
        feature_name = feature_name.replace('CONVERTED', '')
        Path(feature_name[:''.join(feature_name).rindex('/')]).mkdir(parents=True, exist_ok=True)
        print(file_name, " -> ", feature_name)
        pd.DataFrame(feature).to_csv(feature_name)


def create_train_labels(path):
    print("nulla")


if __name__ == '__main__':
    #create_file_of_feature('./timit/**/**/**/*CONVERTED.wav')
    file = open('./timit/test/dr1/faks0/sa1.phn', 'r')
    phoneme = file.read()
    file.close()
    file = open('./preprocessed_dataset/test/dr1/faks0/sa1.csv', 'r')
    feature = pd.read_csv(file)
    file.close()
    phoneme = np.array([[e for e in line.split(' ')] for line in phoneme.splitlines()])
    phoneme[:, 0:2] = (phoneme[:, 0:2].astype(int)) // 16 #convert to ms (sampling rate 16KHz)

    print(feature)
    exit(0)