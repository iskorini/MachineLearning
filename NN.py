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
    model = models.Sequential()
    model.add(
        layers.Dense(120 ,input_shape=(120,), activation='relu')
    )
    model.add(
        layers.Dense(120, activation='relu')
    )
    model.add(
        layers.Dense(3, activation='sigmoid')
    )
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )
    return model

def plot_value(epochs, history):
    plt.figure()
    plt.plot(range(0, epochs), history.history['val_loss'], 'bo', label = "val loss")
    plt.plot(range(0, epochs), history.history['val_acc'], 'b', label="val acc")
    plt.legend()
    plt.show()


def encode_label(label):
    encoder = LabelEncoder()
    encoder.fit(label.astype(str))
    train_encoded_labels = encoder.transform(label)
    return to_categorical(train_encoded_labels)


if __name__ == '__main__':
    phoneme_list = ['p', 't', 'k']
    d = MyDataGenerator('./preprocessed_dataset_LMFE/train/**/**/**.csv')
    train_data, train_label = d.generate_data(phoneme_list)
    d = MyDataGenerator('./preprocessed_dataset_LMFE/test/**/**/**.csv')
    test_data, test_label = d.generate_data(phoneme_list)

    train_label = encode_label(train_label)
    test_label = encode_label(test_label)
    train_data = normalize(train_data, axis=1, order=2)
    test_data = normalize(test_data, axis=1, order=2)
    model = SimpleModel()
    print(model.summary())
    history = model.fit(train_data, train_label, epochs=20, batch_size=64, validation_split=0.3)
    print(model.evaluate(test_data, test_label))
    #exit(0)
