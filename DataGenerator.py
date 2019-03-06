import glob
import pandas as pd
import numpy as np
from itertools import groupby
from operator import itemgetter


class MyDataGenerator:

    def __init__(self, path):
        self.__path = path

    def generate_data(self):
        label = np.empty(0)
        data = np.empty((0,120))
        for file_name in glob.iglob(self.__path):
            file = pd.read_csv(file_name).rename(columns={'Unnamed: 0': 'frame'})
            label = np.concatenate((label, np.array(file.iloc[:, -1])))
            data = np.concatenate((data, np.array(file.iloc[:, 1:121])))
        return data, label





