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

def simple_nn_dense():
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(26,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(61, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def nn_lstm():
    model = models.Sequential()
    model.add(layers.Embedding(61, 61, mask_zero=True, input_length=13885))
    model.add(layers.SimpleRNN(61))
    #model.add(layers.Dense(61, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    train_label = np.empty(0)
    temp_list = []
    L = []
    for file_name in glob.iglob('./preprocessed_dataset/train/dr6/**/*.csv'):
        file = pd.read_csv(file_name)
        file = file.rename(columns={'Unnamed: 0': 'frame'})
        g = file['phoneme'].ne(file['phoneme'].shift()).cumsum()
        L.extend([v for k, v in file.groupby(g)])
        temp_label = np.array(file.iloc[:, -1])
        unique = list(map(itemgetter(0), groupby(temp_label)))
        train_label = np.concatenate((train_label, unique))

    train_data = np.concatenate(temp_list, axis=0)

    encoder = LabelEncoder()
    encoder.fit(train_label.astype(str))
    train_encoded_labels = encoder.transform(train_label)
    train_processed_labels = to_categorical(train_encoded_labels)

    #train_data -= train_data.mean(axis=1)
    #train_data /= train_data.std(axis=1)

    model = nn_lstm()
    print(model.summary())
    history = model.fit(train_data, train_processed_labels,
                        epochs=5, batch_size=256, validation_split=0.2)
    #exit(0)
