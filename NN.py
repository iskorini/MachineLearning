from keras import layers, models
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras import backend as bck
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from DataGenerator import MyDataGenerator
from hyperopt import fmin, tpe, hp, Trials, space_eval
from hyperopt import STATUS_OK
from sklearn.preprocessing import MinMaxScaler


def cnn_spr(train_params):
    model = models.Sequential()
    adam = Adam(lr = train_params['learning_rate'])
    model.add(
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(3, 37, 40))
    )
    model.add(
        layers.MaxPool2D((2, 2), strides=(2,2), data_format='channels_first')
    )
    model.add(
        layers.Conv2D(128, (3, 3), activation='relu')
    )
    model.add(
        layers.MaxPool2D((2,2), strides=(2,2), data_format='channels_first')
    )
    model.add(
        layers.Conv2D(128, (3, 3), padding='same', activation='relu')
    )
    model.add(
        layers.MaxPool2D((2,2), strides=(2,2), data_format='channels_first')
    )
    model.add(
        layers.Flatten()
    )
    model.add(
        layers.Dropout(0.2)
    )
    model.add(
        layers.Dense(1050, activation='relu')
    )
    model.add(
        layers.Dropout(0.2)
    )
    model.add(
        layers.Dense(512, activation='relu')
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
    plt.plot(range(0, ep), history.history['val_loss'], 'r', label = "val loss")
    plt.plot(range(0, ep), history.history['val_acc'], 'b', label="val acc")
    plt.plot(range(0, ep), history.history['loss'], 'g', label = "loss")
    plt.plot(range(0, ep), history.history['acc'], 'm', label="acc")
    plt.plot(ep - 1, ev[0], 'go', label='loss test')
    plt.plot(ep - 1, ev[1], 'mo', label='acc test')
    ax.annotate('%.4f' % (ev[0]), xy=(ep - 2, ev[0] + 0.015) )
    ax.annotate('%.4f' % (ev[1]), xy=(ep - 2, ev[1] + 0.015) )
    ax.annotate('%.4f' % (history.history['acc'][ep - 1]),
                xy=(ep - 2, history.history['acc'][ep - 1] + 0.015))
    ax.annotate('%.4f' % (history.history['loss'][ep - 1]),
                xy=(ep - 2, history.history['loss'][ep - 1] + 0.015))
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.show()


def encode_label(label):
    encoder = LabelEncoder()
    encoder.fit(label.astype(str))
    train_encoded_labels = encoder.transform(label)
    return to_categorical(train_encoded_labels)


def normalize_and_scale_data(data):
    min = data.min(axis=(2, 3), keepdims=True)
    max = data.max(axis=(2, 3), keepdims=True)
    data = (data-min)/(max-min)
    return data


def train_nn(train_d, train_l, ep, params):
    m = cnn_spr(params)
    print(m.summary())
    history = m.fit(train_d, train_l, epochs=ep, batch_size=128, validation_split=0.2)
    return m, history


def eval_nn(model, test_d, test_l):
    return model.evaluate(test_d, test_l)


def opt_nn(params):
    fit_result = train_nn(train_data, train_label, epochs, params)
    test_acc = eval_nn(fit_result[0], test_data, test_label)
    bck.clear_session() #senno smatta ogni bene
    return {'loss': -test_acc[1], 'status': STATUS_OK}


if __name__ == '__main__':
    # data generation
    d_train = MyDataGenerator('./preprocessed_dataset/train/**/*/**.csv')
    d_test = MyDataGenerator('./preprocessed_dataset/test/**/*/**.csv')
    train_data, train_label = d_train.generate_data(phoneme_list=['p', 't', 'k'], max_phn=37)
    test_data, test_label = d_test.generate_data(phoneme_list=['p', 't', 'k'], max_phn=37)
    train_data = normalize_and_scale_data(train_data)
    test_data = normalize_and_scale_data(test_data)
    train_label = encode_label(train_label)
    test_label = encode_label(test_label)
    # train
    epochs = 40
    hp_space_params = {
        'layer_size': hp.choice('layer_size', np.arange(5, 26, 5)),
        'learning_rate': hp.loguniform('learning_rate', -10, 0),
    }
    best = fmin(opt_nn, hp_space_params, algo=tpe.suggest, max_evals=50)
    space_eval = space_eval(hp_space_params, best)
    print(space_eval)
    #fit_result = train_nn(train_data, train_label, epochs, hp_space_params)
    #test_result = eval_nn(fit_result[0], test_data, test_label)
    #plot_value(epochs, fit_result[1], test_result)
    # exit(0)
