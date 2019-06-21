import glob
import pandas as pd
import numpy as np
from math import floor

class MyDataGenerator:

    def __init__(self, path):
        self.__path = path

    def generate_data(self, phoneme_list=None):
        label = np.empty(0)
        data = np.empty((0, 120))
        for file_name in glob.iglob(self.__path):
            data_file = pd.read_csv(file_name)
            if phoneme_list is not None:
                data_file = data_file.loc[data_file['phoneme'].isin(phoneme_list)]
            data = np.concatenate((data,(data_file.iloc[:, 1:121]).to_numpy()))
            label = np.concatenate((label, data_file['phoneme'].to_numpy()))
        return data, label

    def generate_overlapping_chunks(self, timesteps, phoneme_list=None):
        data = pd.DataFrame()
        for file_name in glob.iglob(self.__path):
            data_file = pd.read_csv(file_name)
            if phoneme_list is not None:
                data_file = data_file.loc[data_file['phoneme'].isin(phoneme_list)]
            data = pd.concat((data, data_file))
        data = data.rename(columns={'Unnamed: 0': 'frame'})
        label = np.empty(0)
        data_np = np.empty((1, timesteps, 120))
        for index, row in data.iterrows():
            if index <= data.shape[0]-timesteps:
                c = ((data.iloc[index:(timesteps + index)]).iloc[:, 1:121]).to_numpy().reshape(1, timesteps, 120)
                data_np = np.concatenate((data_np, c))
                label = np.concatenate((label, [data.iloc[index+floor(timesteps/2)]['phoneme']]))
        return data_np[1:], label





