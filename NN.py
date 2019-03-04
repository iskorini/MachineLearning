from keras import layers, models
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import glob
import pandas as pd
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
from operator import itemgetter
from sys import exit
from DataGenerator import MyDataGenerator

def CNN_SPR():
    model = models.Sequential()
    return model


if __name__ == '__main__':

    d = MyDataGenerator('./preprocessed_dataset/train/dr1/**/**.csv')
    train_data, train_label = d.generate_data()

    encoder = LabelEncoder()
    encoder.fit(train_label.astype(str))
    train_encoded_labels = encoder.transform(train_label)
    train_label = to_categorical(train_encoded_labels)

    #train_data -= train_data.mean(axis=1)
    #train_data /= train_data.std(axis=1)

    #model = CNN_SPR()
    #print(model.summary())
    #history = model.fit(train_data, train_label,
    #                    epochs=5, batch_size=256, validation_split=0.2)
    #exit(0)
