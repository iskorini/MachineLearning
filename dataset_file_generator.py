from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from DataGeneratorAll import MyDataGenerator
import numpy as np
import time
import datetime


def encode_label(train, test, val):
    encoder = LabelEncoder()
    encoder.fit(
        np.concatenate(
            (train, np.concatenate((test, val)))
            )
        )
    train_encoded_labels = encoder.transform(train)
    test_encoded_labels = encoder.transform(test)
    val_encoded_labels = encoder.transform(val)
    return to_categorical(train_encoded_labels), to_categorical(test_encoded_labels), to_categorical(val_encoded_labels)


def min_max_scale_skl(train, test, validation):
    tot = np.concatenate((train, validation))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(tot)
    return scaler.transform(train), scaler.transform(test), scaler.transform(validation)


def min_max_scaler_skl_3d(data):
    for i in range(0,data.shape[1]):
        scaler = MinMaxScaler()
        scaler = scaler.fit(data[:, i, :])
        data[:, i, :] = scaler.transform(data[:, i, :])
    return data


if __name__ == '__main__':
    start_time = time.time()
    now = datetime.datetime.now()
    print(str(now.hour) + " " + str(now.minute))
    d = MyDataGenerator('./preprocessed_dataset_LIBROSA/train2/**/**/**.csv')
    train_data, train_label = d.generate_overlapping_chunks_LIBROSA(9)
    d = MyDataGenerator('./preprocessed_dataset_LIBROSA/test/**/**/**.csv')
    test_data, test_label = d.generate_overlapping_chunks_LIBROSA(9)
    d = MyDataGenerator('./preprocessed_dataset_LIBROSA/validation/**/**/**.csv')
    validation_data, validation_label = d.generate_overlapping_chunks_LIBROSA(9)
    train_data, test_data, validation_data = min_max_scale_skl(train_data, test_data, validation_data)
    train_label, test_label, validation_label = encode_label(train_label, test_label, validation_label)
    np.save('./NP_Arrays/RNN/LIBROSA/train_data', train_data)
    np.save('./NP_Arrays/RNN/LIBROSA/test_data', test_data)
    np.save('./NP_Arrays/RNN/LIBROSA/train_label', train_label)
    np.save('./NP_Arrays/RNN/LIBROSA/test_label', test_label)
    np.save('./NP_Arrays/RNN/LIBROSA/validation_label', validation_label)
    np.save('./NP_Arrays/RNN/LIBROSA/validation_data', validation_data)
    now = datetime.datetime.now()
    print(str(now.hour) + " " + str(now.minute))
    print("--- %s seconds ---" % (time.time() - start_time))
    #exit(0)
