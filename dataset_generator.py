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


if __name__ == '__main__':
    phoneme_list = ['p', 't', 'k']
    d = MyDataGenerator('./preprocessed_dataset_LMFE/train/**/**/**.csv')
    train_data, train_label = d.generate_data(phoneme_list)
    d = MyDataGenerator('./preprocessed_dataset_LMFE/test/**/**/**.csv')
    test_data, test_label = d.generate_data(phoneme_list)
    train_label = encode_label(train_label)
    test_label = encode_label(test_label)
    train_data = min_max_scale_skl(train_data)
    test_data = min_max_scale_skl(test_data)
    np.save('./sub_dataset/RNN/train_dataPTK', train_data)
    np.save('./sub_dataset/RNN/test_dataPTK', test_data)
    np.save('./sub_dataset/RNN/train_labelPTK', train_label)
    np.save('./sub_dataset/RNN/test_labelPTK', test_label)
    exit(0)
