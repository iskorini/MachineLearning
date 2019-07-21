import glob
import pandas as pd
import numpy as np
from math import floor
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import datetime


test = '/home/fs6185896/ML_PROJECT/MachineLearning/preprocessed_dataset_LIBROSA/test/dr1/faks0/sa1.csv'
class MyDataGenerator:
    
    def __init__(self, path):
        self.__path = path
        self.__path_train = path[0]
        self.__path_test = path[1]
        self.__path_validation = path[2]

    def generate_overlapping_chunks_LIBROSA(self, timesteps, compact = True):
        data_train = self.generate_data_frame(self.__path_train)
        data_test = self.generate_data_frame(self.__path_test)
        data_validation = self.generate_data_frame(self.__path_validation)
        if compact:
            data_train = self.compact_class(data_train)
            data_test = self.compact_class(data_test)
            data_validation = self.compact_class(data_validation)
        train_n, test_n, validation_n = self.min_max_scale_skl(data_train, data_test, data_validation)
        train_data, train_label = self.generate_chunks(data_train, train_n, timesteps)
        test_data, test_label = self.generate_chunks(data_test, test_n, timesteps)
        validation_data, validation_label = self.generate_chunks(data_validation, validation_n, timesteps)
        train_label, test_label, validation_label = self.encode_label(train_label, test_label, validation_label)
        return train_data, train_label, test_data, test_label, validation_data, validation_label

    def compact_class(self, data_file):
        data_file.loc[data_file['phoneme'] == 'ux', 'phoneme'] = 'uw'
        data_file.loc[data_file['phoneme'] == 'axr', 'phoneme'] = 'er'
        data_file.loc[data_file['phoneme'] == 'em', 'phoneme'] = 'm'
        data_file.loc[data_file['phoneme'] == 'nx', 'phoneme'] = 'n'
        data_file.loc[data_file['phoneme'] == 'eng', 'phoneme'] = 'ng'
        data_file.loc[data_file['phoneme'] == 'hv', 'phoneme'] = 'hh'
        data_file.loc[data_file['phoneme'] == 'h#', 'phoneme'] = 'sil'
        data_file.loc[data_file['phoneme'] == 'pau', 'phoneme'] = 'sil'
        data_file.loc[data_file['phoneme'] == 'pcl', 'phoneme'] = 'sil'
        data_file.loc[data_file['phoneme'] == 'tcl', 'phoneme'] = 'sil'
        data_file.loc[data_file['phoneme'] == 'kcl', 'phoneme'] = 'sil'
        data_file.loc[data_file['phoneme'] == 'bcl', 'phoneme'] = 'sil'
        data_file.loc[data_file['phoneme'] == 'dcl', 'phoneme'] = 'sil'
        data_file.loc[data_file['phoneme'] == 'gcl', 'phoneme'] = 'sil'
        data_file.loc[data_file['phoneme'] == 'epi', 'phoneme'] = 'sil'
        data_file.loc[data_file['phoneme'] == 'zh', 'phoneme'] = 'sh'
        data_file.loc[data_file['phoneme'] == 'en', 'phoneme'] = 'n'
        data_file.loc[data_file['phoneme'] == 'el', 'phoneme'] = 'l'
        data_file.loc[data_file['phoneme'] == 'ix', 'phoneme'] = 'ih'
        data_file.loc[data_file['phoneme'] == 'ax', 'phoneme'] = 'ah'
        data_file.loc[data_file['phoneme'] == 'ax-h', 'phoneme'] = 'ah'
        data_file.loc[data_file['phoneme'] == 'ao', 'phoneme'] = 'aa'
        return data_file

    def generate_data_frame(self, path):
        data = pd.DataFrame()
        for file_name in glob.iglob(path):
            data_file = pd.read_csv(file_name)
            data = pd.concat((data, data_file))
        data = data.rename(columns={'Unnamed: 0': 'frame'}) 
        return data
    
    def min_max_scale_skl(self, train, test, validation):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(np.concatenate((train.iloc[:, 1:124], test.iloc[:, 1:124], validation.iloc[:, 1:124])))
        return scaler.transform(train.iloc[:, 1:124]), scaler.transform(test.iloc[:, 1:124]), scaler.transform(validation.iloc[:, 1:124])

    def generate_chunks(self, data, data_norm, timesteps):
        label = np.empty(0)
        data_np = np.empty((1, timesteps, 123))
        for index, row in data.iterrows():
            if index <= data.shape[0]-timesteps:
                c = ((data_norm[index:(timesteps + index)])).reshape(1, timesteps, (124-1))
                data_np = np.concatenate((data_np, c))
                label = np.concatenate((label, [data.iloc[index+floor(timesteps/2)]['phoneme']]))
        return data_np[1:], label

    def encode_label(self, train, test, val):
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

if __name__ == "__main__":
    start_time = time.time()
    now = datetime.datetime.now()
    print(str(now.hour) + " " + str(now.minute))
    path = ['./preprocessed_dataset_LIBROSA/train2/**/**/**.csv', './preprocessed_dataset_LIBROSA/test/**/**/**.csv',
    './preprocessed_dataset_LIBROSA/validation/**/**/**.csv']
    d = MyDataGenerator(path)
    train_data, train_label, test_data, test_label, validation_data, validation_label = d.generate_overlapping_chunks_LIBROSA(3)
    np.save('./NP_Arrays/RNN/LIBROSA/train_data_3_COMPLETE', train_data)
    np.save('./NP_Arrays/RNN/LIBROSA/test_data_3_COMPLETE', test_data)
    np.save('./NP_Arrays/RNN/LIBROSA/train_label_3_COMPLETE', train_label)
    np.save('./NP_Arrays/RNN/LIBROSA/test_label_3_COMPLETE', test_label)
    np.save('./NP_Arrays/RNN/LIBROSA/validation_label_3_COMPLETE', validation_label)
    np.save('./NP_Arrays/RNN/LIBROSA/validation_data_3_COMPLETE', validation_data)
    now = datetime.datetime.now()
    print(str(now.hour) + " " + str(now.minute))
    print("--- %s seconds ---" % (time.time() - start_time))