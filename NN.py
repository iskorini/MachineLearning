import os
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";

from keras import layers, models
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical
from keras import backend as bck
from keras.utils import multi_gpu_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from DataGenerator import MyDataGenerator
from hyperopt import fmin, tpe, hp, Trials, space_eval
from hyperopt import STATUS_OK
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample

from sklearn.preprocessing import MinMaxScaler


def cnn_spr(train_params):
    model = models.Sequential()
    if train_params['optimizer']['name'] == 'adam':
        opt = Adam(lr=train_params['optimizer']['learning_rate'])
    else:
        opt = SGD(lr=train_params['optimizer']['learning_rate'], momentum=train_params['optimizer']['momentum'])
    model.add(
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(3, 37, 40))
    )
    model.add(
        layers.MaxPool2D((2, 2), strides=(2, 2), data_format='channels_first')
    )
    model.add(
        layers.Conv2D(128, (3, 3), activation='relu')
    )
    model.add(
        layers.MaxPool2D((2, 2), strides=(2, 2), data_format='channels_first')
    )
    model.add(
        layers.Conv2D(128, (3, 3), padding='same', activation='relu')
    )
    model.add(
        layers.MaxPool2D((2, 2), strides=(2, 2), data_format='channels_first')
    )
    model.add(
        layers.Flatten()
    )
    model.add(
        layers.Dropout(train_params['dropout_variation1'])
    )
    model.add(
        layers.Dense(int(train_params['layer_size1']), activation='relu')
    )
    model.add(
        layers.Dropout(train_params['dropout_variation2'])
    )
    model.add(
        layers.Dense(int(train_params['layer_size2']), activation='relu')
    )
    model.add(
        layers.Dense(3, activation='sigmoid')
    )
    # try:
    #	model = multi_gpu_model(model)
    # except:
    #    pass
    #    print("SINGOLA GPU")
    #    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
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


def encode_label(label):
    encoder = LabelEncoder()
    encoder.fit(label.astype(str))
    train_encoded_labels = encoder.transform(label)
    return to_categorical(train_encoded_labels)


def normalize_and_scale_data(data):
    min = data.min(axis=(2, 3), keepdims=True)
    max = data.max(axis=(2, 3), keepdims=True)
    data = (data - min) / (max - min)
    return data


def train_nn(train_d, train_l, ep, params):
    m = cnn_spr(params)
    print(m.summary())
    history = m.fit(train_d, train_l, epochs=ep, batch_size=int(params['batch_size']), validation_split=0.2)
    return m, history


def eval_nn(model, test_d, test_l):
    return model.evaluate(test_d, test_l)


def opt_nn(params):
    fit_result = train_nn(train_data, train_label, epochs, params)
    test_acc = eval_nn(fit_result[0], test_data, test_label)
    # bck.clear_session()
    return {'loss': -test_acc[1], 'status': STATUS_OK}


@scope.define
def power_of_two(a):
    return 2.0 ** a


if __name__ == '__main__':
    # data generation
    # d_train = MyDataGenerator('./preprocessed_dataset/train/**/*/**.csv')
    # d_test = MyDataGenerator('./preprocessed_dataset/test/**/*/**.csv')
    # train_data, train_label = d_train.generate_data(phoneme_list=['p', 't', 'k'], max_phn=37)
    # test_data, test_label = d_test.generate_data(phoneme_list=['p', 't', 'k'], max_phn=37)
    # train_data = normalize_and_scale_data(train_data)
    # test_data = normalize_and_scale_data(test_data)
    # train_label = encode_label(train_label)
    # test_label = encode_label(test_label)
    # load data
    test_data = np.load("./NP_Arrays/test_data_ptk.arr.npy")
    test_label = np.load("./NP_Arrays/test_label_ptk.arr.npy")
    train_data = np.load("./NP_Arrays/train_data_ptk.arr.npy")
    train_label = np.load("./NP_Arrays/train_label_ptk.arr.npy")
    # train
    epochs = 100
    opt_search_space = hp.choice('name',
                                 [
                                     {'name': 'adam',
                                      'learning_rate': hp.loguniform('learning_rate_adam', -10, 0),
                                      # Note the name of the label to avoid duplicates
                                      },
                                     {'name': 'sgd',
                                      'learning_rate': hp.loguniform('learning_rate_sgd', -15, 1),
                                      # Note the name of the label to avoid duplicates
                                      'momentum': hp.uniform('momentum', 0, 1.0),
                                      }
                                 ])
    hp_space_params = {
        'optimizer': opt_search_space,
        'dropout_variation1': hp.quniform('dropout_variation1', 0.2, 0.6, 0.1),
        'dropout_variation2': hp.quniform('dropout_variation2', 0.2, 0.6, 0.1),
        'layer_size1': hp.quniform('layer_size1', low=500, high=2000, q=200),
        'layer_size2': hp.quniform('layer_size2', low=250, high=1000, q=100),
        'batch_size': scope.power_of_two(hp.quniform('batch_size', 0, 8, q=1))
    }
    #trials = Trials()
    #best = fmin(opt_nn, hp_space_params, algo=tpe.suggest, max_evals=50, trials=trials)
    #space_eval = space_eval(hp_space_params, best)
    #pickle.dump(trials, open("trials.p", "wb"))
    #print(space_eval)

    params = {
        'optimizer': {
            'name':'adam',
            'learning_rate': 0.00015221574430693675},
        'dropout_variation1': 0.3,
        'dropout_variation2': 0.2,
        'layer_size1': 800,
        'layer_size2': 1000,
        'batch_size': 4
    }
    fit_result = train_nn(train_data, train_label, epochs, params)
    test_result = eval_nn(fit_result[0], test_data, test_label)
    plot_value(epochs, fit_result[1], test_result)
    exit(0)
