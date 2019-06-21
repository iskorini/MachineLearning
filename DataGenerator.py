import glob
import pandas as pd
import numpy as np


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




