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
            data = np.concatenate((data, (data_file.iloc[:, 1:121]).to_numpy()))
            label = np.concatenate((label, data_file['phoneme'].to_numpy()))
        return data, label

    def generate_ALL_data_MFCC(self):
        data = pd.DataFrame()
        for file_name in glob.iglob(self.__path):
            data_file = pd.read_csv(file_name)
            ############## CLASS REDUCTION
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
            ##############
            data = pd.concat((data, data_file))
        data = data.rename(columns={'Unnamed: 0': 'frame'})
        return data.iloc[:, 1:40].to_numpy(), data.iloc[:, 42].to_numpy()  # data, label

    def generate_overlapping_chunks_LMFE(self, timesteps, phoneme_list=None):
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
            if index <= data.shape[0] - timesteps:
                c = ((data.iloc[index:(timesteps + index)]).iloc[:, 1:121]).to_numpy().reshape(1, timesteps, 120)
                data_np = np.concatenate((data_np, c))
                label = np.concatenate((label, [data.iloc[index + floor(timesteps / 2)]['phoneme']]))
        return data_np[1:], label

    def generate_overlapping_chunks_MFCC(self, timesteps, phoneme_list=None):
        data = pd.DataFrame()
        for file_name in glob.iglob(self.__path):
            data_file = pd.read_csv(file_name)
            if phoneme_list is not None:
                data_file = data_file.loc[data_file['phoneme'].isin(phoneme_list)]
            ##############
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
            ##############
            data = pd.concat((data, data_file))
        data = data.rename(columns={'Unnamed: 0': 'frame'})
        label = np.empty(0)
        data_np = np.empty((1, timesteps, 39))
        for index, row in data.iterrows():
            if index <= data.shape[0] - timesteps:
                c = ((data.iloc[index:(timesteps + index)]).iloc[:, 1:40]).to_numpy().reshape(1, timesteps, 39)
                data_np = np.concatenate((data_np, c))
                label = np.concatenate((label, [data.iloc[index + floor(timesteps / 2)]['phoneme']]))
        return data_np[1:], label





