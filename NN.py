from keras import layers, models
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_data = np.empty((0, 26))
    train_label = np.empty(0)
    for file_name in glob.iglob('./preprocessed_dataset/train/**/**/*.csv'):
        file = pd.read_csv(file_name)
        file = file.rename(columns={'Unnamed: 0': 'frame'})
        file.loc[0, 'phoneme'] = 'h#'  # LOL
        temp_data = np.array(file.iloc[:, 1:27])
        temp_label = np.array(file.iloc[:, -1])
        train_data = np.concatenate((train_data, temp_data), axis=0)
        train_label = np.concatenate((train_label, temp_label))

    encoder = LabelEncoder()
    encoder.fit(train_label.astype(str))
    encoded_labels = encoder.transform(train_label)
    processed_labels = to_categorical(encoded_labels)
    train_data -= train_data.mean(axis=0)
    train_data /= train_data.std(axis=0)
    epochs = 50

    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(26,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(61, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics = ['accuracy'])

    history = model.fit(train_data, processed_labels, epochs=epochs, batch_size=128)
    plt.plot(epochs, history.history['acc'], 'bo', label = 'accuracy')
    plt.plot(epochs, history.history['loss'], label = 'loss')
    plt.legend()
    plt.show()
    exit(0)
