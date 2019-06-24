from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from DataGenerator import MyDataGenerator
import numpy as np


def encode_label(label):
    encoder = LabelEncoder()
    encoder.fit(label.astype(str))
    train_encoded_labels = encoder.transform(label)
    return to_categorical(train_encoded_labels)


def min_max_scale_skl(data):
    scaler = MinMaxScaler()
    scaler = scaler.fit(data)
    return scaler.transform(data)


def min_max_scaler_skl_3d(data):
    for i in range(0,data.shape[1]):
        scaler = MinMaxScaler()
        scaler = scaler.fit(data[:, i, :])
        data[:, i, :] = scaler.transform(data[:, i, :])
    return data


if __name__ == '__main__':
    phoneme_list = ['p', 't', 'k']
    d = MyDataGenerator('./preprocessed_dataset_LMFE/train/**/**/**.csv')
    train_data, train_label = d.generate_overlapping_chunks(10, phoneme_list)
    d = MyDataGenerator('./preprocessed_dataset_LMFE/test/**/**/**.csv')
    test_data, test_label = d.generate_overlapping_chunks(10, phoneme_list)
    train_label = encode_label(train_label)
    test_label = encode_label(test_label)
    #train_data = min_max_scaler_skl_3d(train_data)
    #test_data = min_max_scaler_skl_3d(test_data)
    np.save('./NP_Arrays/RNN/train_dataPTK_10_NO_SCALE', train_data)
    np.save('./NP_Arrays/RNN/test_dataPTK_10_NO_SCALE', test_data)
    np.save('./NP_Arrays/RNN/train_labelPTK_10_NO_SCALE', train_label)
    np.save('./NP_Arrays/RNN/test_labelPTK_10_NO_SCALE', test_label)
    #exit(0)
