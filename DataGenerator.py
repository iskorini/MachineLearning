import glob
import pandas as pd
import numpy as np


class MyDataGenerator:

    def __init__(self, path):
        self.__path = path

    def generate_data(self, phoneme_list=None):
        label = np.empty(0)
        data = np.empty((0,120))
        for file_name in glob.iglob(self.__path):
            file = pd.read_csv(file_name).rename(columns={'Unnamed: 0': 'frame'})
            if phoneme_list is not None:
                file = file.loc[file['phoneme'].isin(phoneme_list)]
            label = np.concatenate((label, np.array(file.iloc[:, -1])))
            data = np.concatenate((data, np.array(file.iloc[:, 1:121])))
        return data, label





