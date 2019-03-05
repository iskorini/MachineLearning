from keras import layers, models
from keras.utils.np_utils import to_categorical
from keras.utils import normalize
from sklearn.preprocessing import LabelEncoder
import glob
import pandas as pd
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
from operator import itemgetter
from sys import exit
from DataGenerator import MyDataGenerator

def SimpleModel():

def encode_label(label):
    encoder = LabelEncoder()
    encoder.fit(train_label.astype(str))
    train_encoded_labels = encoder.transform(train_label)
    return to_categorical(train_encoded_labels)


if __name__ == '__main__':

    d = MyDataGenerator('./preprocessed_dataset/train/dr1/**/**.csv')
    train_data, train_label = d.generate_data()

    train_label = encode_label(train_label)

    train_data = normalize(train_data, axis=1, order=2)

    model = CNN_SPR()
    print(model.summary())
    history = model.fit(train_data, train_label, epochs=5, batch_size=128, validation_split=0.2)
    #exit(0)
