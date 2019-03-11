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
    model.add(
        layers.Conv2D(150, kernel_size=3, activation='relu', input_shape=(3, 465, 40))
    )
    model.add(
        layers.MaxPool2D(pool_size=2, data_format='channels_first')
    )
    model.add(layers.Flatten())
    model.add(
        layers.Dense(3, activation='softmax')
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['acc']
    )
    return model

def plot_value(epochs, history):
    plt.figure()
    plt.plot(range(0, epochs), history.history['val_loss'], 'r', label = "val loss")
    plt.plot(range(0, epochs), history.history['val_acc'], 'b', label="val acc")
    plt.legend()
    plt.show()


def encode_label(label):
    encoder = LabelEncoder()
    encoder.fit(label.astype(str))
    train_encoded_labels = encoder.transform(label)
    return to_categorical(train_encoded_labels)

if __name__ == '__main__':

    d_train = MyDataGenerator('./preprocessed_dataset/train/**/*/**.csv')
    d_test = MyDataGenerator('./preprocessed_dataset/test/**/*/**.csv')
    train_data, train_label = d_train.generate_data(phoneme_list=['p', 't', 'k'], max_phn=465)
    test_data, test_label = d_test.generate_data(phoneme_list=['p', 't', 'k'], max_phn=465)
    train_label = encode_label(train_label)
    test_label = encode_label(test_label)

    model = CNN_SPR()
    print(model.summary())
    history = model.fit(train_data, train_label,
                        epochs=20, batch_size=128, validation_split=0.2)
    print(model.evaluate(test_data, test_label))

    #exit(0)
