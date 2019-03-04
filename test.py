from keras import layers, models
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import glob
import pandas as pd
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
from operator import itemgetter


file1 = pd.read_csv('./preprocessed_dataset/test/dr1/faks0/sx43.csv')
file2 = pd.read_csv('./preprocessed_dataset/test/dr1/faks0/sa1.csv')
file3 = pd.read_csv('./preprocessed_dataset/test/dr1/faks0/sa2.csv')

g1 = file1['phoneme'].ne(file1['phoneme'].shift()).cumsum()
g2 = file2['phoneme'].ne(file2['phoneme'].shift()).cumsum()
g3 = file3['phoneme'].ne(file3['phoneme'].shift()).cumsum()

L1 = [v.iloc[:, 1:27] for k, v in file1.groupby(g1)]

L2 = [v.iloc[:, 1:27] for k, v in file2.groupby(g2)]

L3 = [v.iloc[:, 1:27] for k, v in file3.groupby(g3)]
data = np.array((0, 0, 26))

data1 = np.array(list(map(lambda x: np.matrix(x), L1)))
data2 = np.array(list(map(lambda x: np.matrix(x), L2)))
data3 = np.array(list(map(lambda x: np.matrix(x), L3)))

list = []
list.append(data1)
list.append(data2)
list.append(data3)
print(data1.shape)
print(data2.shape)
print(data3.shape)
print(len(list))
a = np.concatenate(list, axis=0)
print(a.shape)