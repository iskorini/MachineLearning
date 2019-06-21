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

def plot_value(ep, history, ev):
    fig, ax = plt.subplots()
    plt.plot(range(0, ep), history.history['val_loss'], 'r', label="val loss")
    plt.plot(range(0, ep), history.history['val_acc'], 'b', label="val acc")
    plt.plot(range(0, ep), history.history['loss'], 'g', label="loss")
    plt.plot(range(0, ep), history.history['acc'], 'm', label="acc")
    plt.plot(ep - 1, ev[0], 'go', label='loss test')
    plt.plot(ep - 1, ev[1], 'mo', label='acc test')
    ax.annotate('%.4f' % (ev[0]), xy=(ep - 2, ev[0] + 0.015))
    ax.annotate('%.4f' % (ev[1]), xy=(ep - 2, ev[1] + 0.015))
    ax.annotate('%.4f' % (history.history['acc'][ep - 1]),
                xy=(ep - 2, history.history['acc'][ep - 1] + 0.015))
    ax.annotate('%.4f' % (history.history['loss'][ep - 1]),
                xy=(ep - 2, history.history['loss'][ep - 1] + 0.015))
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.show()


if __name__ == '__main__':
    phoneme_list = ['p', 't', 'k']
    d = MyDataGenerator('./preprocessed_dataset_LMFE/train/**/**/**.csv')
    train_data, train_label = d.generate_data(phoneme_list)
    d = MyDataGenerator('./preprocessed_dataset_LMFE/test/**/**/**.csv')
    test_data, test_label = d.generate_data(phoneme_list)
    train_label = encode_label(train_label)
    test_label = encode_label(test_label)
    #train_data = normalize_and_scale_data(train_data)
    #test_data = normalize_and_scale_data(test_data)
    #model = SimpleModel()
    #print(model.summary())
    #history = model.fit(train_data, train_label, epochs=20, batch_size=64, validation_split=0.3)
    #print(model.evaluate(test_data, test_label))
    #exit(0)
