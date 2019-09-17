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
import argparse
import os

test = '/home/fs6185896/ML_PROJECT/MachineLearning/preprocessed_dataset_LIBROSA/test/dr1/faks0/sa1.csv'
class MyDataGenerator:
    
    def __init__(self, path, args):
        self.__path_train = path[0]
        self.__path_test = path[1]
        self.__path_validation = path[2]
        self.__compact = args.ext
        self.__timesteps = args.c
        self.__normalizing = args.n

    def start(self):
        paths = [self.__path_train, self.__path_test, self.__path_validation]
        pool = ThreadPool(3)
        results = pool.map(self.generate_data_frame, paths)
        pool.close()
        pool.join()
        data_train = results[0]
        data_test = results[1]
        data_validation = results[2]
        if not self.__compact:
            data_train = self.compact_class(data_train)
            data_test = self.compact_class(data_test)
            data_validation = self.compact_class(data_validation)
        train_n, test_n, validation_n = self.min_max_scale_skl(data_train, data_test, data_validation)
        dim = int(train_n.shape[0]/12)

        args1 = [data_train.iloc[0:dim,:], data_train.iloc[dim:dim*2,:], data_train.iloc[(dim*2):dim*3,:]
        , data_train.iloc[(dim*3):dim*4,:], data_train.iloc[(dim*4):dim*5,:], data_train.iloc[(dim*5):dim*6,:],
        data_train.iloc[dim*6:dim*7,:],data_train.iloc[dim*7:dim*8,:],data_train.iloc[dim*8:dim*9,:],data_train.iloc[dim*9:10,:],
        data_train.iloc[dim*10:dim*11,:], data_train.iloc[dim*11:-1,:],

        data_test.iloc[0:dim,:], data_test.iloc[dim:dim*2,:], data_test.iloc[dim*2:dim*3,:],  data_test.iloc[dim*3:-1,:],
        data_validation.iloc[0:dim,:], data_validation.iloc[dim:dim*2,:], data_validation.iloc[dim*2:dim*3,:],  data_validation.iloc[dim*3:-1,:]]
        
        args2 = [train_n[0:dim,:], train_n[dim:dim*2,:], train_n[(dim*2):dim*3,:]
        , train_n[(dim*3):dim*4,:], train_n[(dim*4):dim*5,:], train_n[(dim*5):dim*6,:],
        train_n[dim*6:dim*7,:],train_n[dim*7:dim*8,:],train_n[dim*8:dim*9,:],train_n[dim*9:10,:],
        train_n[dim*10:dim*11,:], train_n[dim*11:-1,:],

        test_n[0:dim,:], test_n[dim:dim*2,:], test_n[dim*2:dim*3,:],  test_n[dim*3:-1,:],
        validation_n[0:dim,:], validation_n[dim:dim*2,:], validation_n[dim*2:dim*3,:],  validation_n[dim*3:-1,:]]
    
        pool = ThreadPool(12)
        results = pool.starmap(self.generate_chunks, zip(args1, args2, itertools.repeat(self.__timesteps)))
        pool.close()
        pool.join()

        train_data_set = np.concatenate((results[0][0], results[1][0], results[2][0], results[3][0], results[4][0], results[5][0],
        results[6][0], results[7][0], results[8][0], results[9][0], results[10][0], results[11][0]))
        test_data_set = np.concatenate((results[12][0], results[13][0], results[14][0], results[15][0]))
        validation_data_set = np.concatenate((results[16][0], results[17][0], results[18][0], results[19][0]))

        train_label = np.concatenate((results[0][1], results[1][1], results[2][1], results[3][1], results[4][1], results[5][1],
        results[6][1], results[7][1], results[8][1], results[9][1], results[10][1], results[11][1]))
        test_label = np.concatenate((results[12][1], results[13][1], results[14][1], results[15][1]))
        validation_label = np.concatenate((results[16][1], results[17][1], results[18][1], results[19][1]))
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
        if not self.__normalizing:
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
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
            data_np.append(data_norm[i:b[i]].reshape(1, timesteps, (124 - 1)))
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
    data_path = ['./preprocessed_dataset_LIBROSA/train2/**/**/**.csv', './preprocessed_dataset_LIBROSA/test/**/**/**.csv',
    './preprocessed_dataset_LIBROSA/validation/**/**/**.csv']
    parser = argparse.ArgumentParser(description='Generate some data')
    parser.add_argument('--c', help = 'sliding window', type=int)
    parser.add_argument('--n', help = 'normalizing -1 1, default 0 1', default=False, action='store_true')
    parser.add_argument('--ext', help='if used 61 class, default 40', default=False, action='store_true')
    arguments = parser.parse_args()
    print(arguments)
    name_folder = 'W'+str(arguments.c)+'_'
    if not arguments.ext:
        name_folder = name_folder+'40_'
    else:
        name_folder = name_folder+'61_'
    if not arguments.n:
        name_folder = name_folder+'01'
    else:
        name_folder = name_folder+'11'
    path = './NP_Arrays/RNN/LIBROSA/'+name_folder
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)
    d = MyDataGenerator(data_path, arguments)
    print(path+'/train_data')
    print(path+'/test_data')
    print(path+'/train_label')
    print(path+'/test_label')
    print(path+'/validation_label')
    print(path+'/validation_data')
    train_data, train_label, test_data, test_label, validation_data, validation_label = d.start()
    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)
    print(validation_data.shape)
    print(validation_label.shape)
    np.save(path+'/train_data', train_data)
    np.save(path+'/test_data', test_data)
    np.save(path+'/train_label', train_label)
    np.save(path+'/test_label', test_label)
    np.save(path+'/validation_label', validation_label)
    np.save(path+'/validation_data', validation_data)
    now = datetime.datetime.now()
    print(str(now.hour) + " " + str(now.minute))
    print("--- %s seconds ---" % (time.time() - start_time))