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
import progressbar
import itertools
from multiprocessing.pool import Pool as ThreadPool

test = '/home/fs6185896/ML_PROJECT/MachineLearning/preprocessed_dataset_LIBROSA/test/dr1/faks0/sa1.csv'
class MyDataGenerator:
    
    def __init__(self, path):
        self.__path = path
        self.__path_train = path[0]
        self.__path_test = path[1]
        self.__path_validation = path[2]

    def generate_overlapping_chunks_LIBROSA(self, timesteps, compact = True):
        args = [self.__path_train, self.__path_test, self.__path_validation]
        pool = ThreadPool(3)
        results = pool.map(self.generate_data_frame, args)
        pool.close()
        pool.join()
        data_train = results[0]
        data_test = results[1]
        data_validation = results[2]
        if compact:
            data_train = self.compact_class(data_train)
            data_test = self.compact_class(data_test)
            data_validation = self.compact_class(data_validation)
        train_n, test_n, validation_n = self.min_max_scale_skl(data_train, data_test, data_validation)
        dim = int(train_n.shape[0]/6)
        #train
        args1 = [data_train.iloc[0:dim,:], data_train.iloc[dim:dim*2,:], data_train.iloc[(dim*2):dim*3,:]
        , data_train.iloc[(dim*3):dim*4,:], data_train.iloc[(dim*4):dim*5,:], data_train.iloc[(dim*5):-1,:],
         data_test.iloc[0:dim,:], data_test.iloc[dim:-1,:], data_validation.iloc[0:dim,:], data_validation.iloc[dim:-1,:]]
        args2 = [train_n[0:dim], train_n[dim:dim*2], train_n[(dim*2):dim*3], train_n[(dim*3):dim*4], train_n[(dim*4):dim*5]
        , train_n[(dim*5):-1], test_n[0:dim] , test_n[dim:-1,:], validation_n[0:dim,:], validation_n[dim:-1,:]]
        pool = ThreadPool(8)
        results = pool.starmap(self.generate_chunks, zip(args1, args2, itertools.repeat(timesteps)))
        pool.close()
        pool.join()

        train_data_set = np.concatenate((results[0][0], results[1][0], results[2][0], results[3][0], results[4][0], results[5][0]))
        test_data_set = np.concatenate((results[6][0], results[7][0]))
        validation_data_set = np.concatenate((results[8][0], results[9][0]))

        train_label = np.concatenate((results[0][1], results[1][1], results[2][1], results[3][1], results[4][1], results[5][1]))
        test_label = np.concatenate((results[6][1], results[7][1]))
        validation_label = np.concatenate((results[8][1], results[9][1]))
        train_label, test_label, validation_label = self.encode_label(train_label, test_label, validation_label)
        return train_data_set, train_label, test_data_set,test_label , validation_data_set , validation_label
        
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
        tot = len(glob.glob(path))
        bar = progressbar.ProgressBar(maxval=tot, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        i = 0
        bar.start()
        data = []
        for file_name in glob.iglob(path):
            data.append(pd.read_csv(file_name))
            i = i+1
            bar.update(i)
        data = pd.concat(data)
        bar.finish()
        data = data.rename(columns={'Unnamed: 0': 'frame'}) 
        return data
    
    def min_max_scale_skl(self, train, test, validation):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(np.concatenate((train.iloc[:, 1:124], validation.iloc[:, 1:124])))
        return scaler.transform(train.iloc[:, 1:124]), scaler.transform(test.iloc[:, 1:124]), scaler.transform(validation.iloc[:, 1:124])

    def generate_chunks(self, data, data_norm, timesteps):
        label = []
        data_np = []
        b = range(timesteps, data.shape[0]+1) 
        bar = progressbar.ProgressBar(maxval=data.shape[0]-timesteps, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i in range(0, data.shape[0]-timesteps+1):
            data_np.append(data_norm[i:b[i]].reshape(1, (124 - 1)*3))
            label.append(data.iloc[i + floor(timesteps / 2)]['phoneme'])
            bar.update(i)
        data_np = np.concatenate(data_np)
        label = np.array(label)
        bar.finish()
        return data_np, label

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
    np.save('./NP_Arrays/RNN/LIBROSA/train_data', train_data)
    np.save('./NP_Arrays/RNN/LIBROSA/test_data', test_data)
    np.save('./NP_Arrays/RNN/LIBROSA/train_label', train_label)
    np.save('./NP_Arrays/RNN/LIBROSA/test_label', test_label)
    np.save('./NP_Arrays/RNN/LIBROSA/validation_label', validation_label)
    np.save('./NP_Arrays/RNN/LIBROSA/validation_data', validation_data)
    now = datetime.datetime.now()
    print(str(now.hour) + " " + str(now.minute))
    print("--- %s seconds ---" % (time.time() - start_time))