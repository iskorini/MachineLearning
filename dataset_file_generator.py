from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from DataGenerator import MyDataGenerator
import numpy as np
import time

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
    start_time = time.time()
    phoneme_list = ['p', 't', 'k']
    d = MyDataGenerator('./preprocessed_dataset_MFCC/train/dr1/**/**.csv')
    train_data, train_label = d.generate_overlapping_chunks_MFCC(20, phoneme_list)
    d = MyDataGenerator('./preprocessed_dataset_MFCC/test/dr1/**/**.csv')
    test_data, test_label = d.generate_overlapping_chunks_MFCC(20, phoneme_list)
    #d = MyDataGenerator('./preprocessed_dataset_MFCC/validation/**/**/**.csv')
    #validation_data, validation_label = d.generate_overlapping_chunks_MFCC(20, phoneme_list)
    train_label = encode_label(train_label)
    test_label = encode_label(test_label)
    #validation_label = encode_label(validation_label)
    train_data = min_max_scaler_skl_3d(train_data)
    test_data = min_max_scaler_skl_3d(test_data)
    #validation_data = min_max_scaler_skl_3d(validation_data)
    np.save('./NP_Arrays/RNN/MFCC/train_dataPTK_20_DR1', train_data)
    np.save('./NP_Arrays/RNN/MFCC/test_dataPTK_20_DR1', test_data)
    np.save('./NP_Arrays/RNN/MFCC/train_labelPTK_20_DR1', train_label)
    np.save('./NP_Arrays/RNN/MFCC/test_labelPTK_20_DR1', test_label)
    #np.save('./NP_Arrays/RNN/MFCC/validation_labelPTK_20', validation_label)
    #np.save('./NP_Arrays/RNN/MFCC/validation_dataPTK_20', validation_data)
    print("--- %s seconds ---" % (time.time() - start_time))
    #exit(0)
