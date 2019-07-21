import os
import pickle
from keras import layers, models
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical
from keras import backend as bck
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from DataGenerator import MyDataGenerator
from hyperopt import fmin, tpe, hp, Trials, space_eval
from hyperopt import STATUS_OK
from hyperopt.pyll import scope

from sklearn.preprocessing import MinMaxScaler

def encode_label(label):
    encoder = LabelEncoder()
    encoder.fit(label.astype(str))
    train_encoded_labels = encoder.transform(label)
    return to_categorical(train_encoded_labels)


def normalize_and_scale_data(data):
    min = data.min(axis=(2, 3), keepdims=True)
    max = data.max(axis=(2, 3), keepdims=True)
    data = (data - min) / (max - min)
    return data



if __name__ == "__main__":
    # data generation
    d_train = MyDataGenerator('./preprocessed_dataset/train/**/*/**.csv')
    d_test = MyDataGenerator('./preprocessed_dataset/test/**/*/**.csv')
    print("start read: train_data, train_label")
    train_data, train_label = d_train.generate_data(phoneme_list=['p', 't', 'k'], max_phn=37)
    print("end read: train_data, train_label")
    print("start read: test_data, test_label")
    test_data, test_label = d_test.generate_data(phoneme_list=['p', 't', 'k'], max_phn=37)
    print("end read: test_data, test_label")
    print("start normalize: train_data")
    train_data = normalize_and_scale_data(train_data)
    print("end normalize: train_data")
    print("start normalize: test_data")
    test_data = normalize_and_scale_data(test_data)
    print("end normalize: test_data")
    print("start encoding: train_label")
    train_label = encode_label(train_label)
    print("end encoding: train_label")
    print("start encoding: test_label")
    test_label = encode_label(test_label)
    print("end encoding: test_label")
    print("saving: test_data")
    np.save('./NP_Arrays/test_data_ptk.arr', test_data)
    print("saved")
    print("saving: train_data")
    np.save('./NP_Arrays/train_data_ptk.arr', train_data)
    print("saved")
    print("saving: test_label")
    np.save('./NP_Arrays/test_label_ptk.arr', test_label)
    print("saved")
    print("saving: train_label")
    np.save('./NP_Arrays/train_label_ptk.arr', train_label)
    print("saved")
    print("DONE.")