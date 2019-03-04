import glob
import pandas as pd
import numpy as np
from itertools import groupby
from operator import itemgetter


class MyDataGenerator:

    def __init__(self, path):
        self.__path = path

    def generate_data(self):
        L0 = []
        L1 = []
        L2 = []
        label = np.empty(0)
        for file_name in glob.iglob(self.__path):
            file_name = file_name.replace('preprocessed_dataset', 'preprocessed_dataset_LMFE')
            file0 = pd.read_csv(file_name.replace('.csv', '-0.csv'))
            file1 = pd.read_csv(file_name.replace('.csv', '-1.csv'))
            file2 = pd.read_csv(file_name.replace('.csv', '-2.csv'))
            file1 = file1.rename(columns={'Unnamed: 0': 'frame'})  # optional
            file0 = file0.rename(columns={'Unnamed: 0': 'frame'})  # optional
            file2 = file2.rename(columns={'Unnamed: 0': 'frame'})  # optional
            g0 = file0['phoneme'].ne(file0['phoneme'].shift()).cumsum()
            g1 = file1['phoneme'].ne(file1['phoneme'].shift()).cumsum()
            g2 = file2['phoneme'].ne(file2['phoneme'].shift()).cumsum()
            L0.extend([v for k, v in file0.groupby(g0)])
            L1.extend([v for k, v in file1.groupby(g1)])
            L2.extend([v for k, v in file2.groupby(g2)])
            temp_label = np.array(file0.iloc[:, -1])
            unique = list(map(itemgetter(0), groupby(temp_label)))
            label = np.concatenate((label, unique))
        max_l = len(max(L0, key=len))
        data = np.empty((len(L0), 3, max_l, 40))
        for i in range(0, len(L0)):
            static = L0[i].iloc[:, 1:41].to_numpy()
            first = L1[i].iloc[:, 1:41].to_numpy()
            second = L2[i].iloc[:, 1:41].to_numpy()
            diff = max_l - static.shape[0]
            static = np.pad(static, ((0, diff), (0, 0)), mode='constant', constant_values=0)
            first = np.pad(first, ((0, diff), (0, 0)), mode='constant', constant_values=0)
            second = np.pad(second, ((0, diff), (0, 0)), mode='constant', constant_values=0)
            data[i] = np.array([static, first, second])
        return data, label





