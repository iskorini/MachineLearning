import os
from time import time
from keras import layers, models, optimizers
from keras.callbacks import TensorBoard, TerminateOnNaN
import matplotlib.pyplot as plt
import numpy as np


def RNN_model(params):
    model = models.Sequential()
    model.add(
        layers.TimeDistributed(layers.Dense(params['layer_size1'], activation='relu'), input_shape=(10, 120))
    )
    model.add(
        layers.LSTM(params['RNN_SIZE1'], dropout=params['dropout'],
                    recurrent_dropout=params['recurrent_dropout'],
                    activation='relu', return_sequences=True)
    )
    model.add(
        layers.LSTM(params['RNN_SIZE2'], dropout=params['dropout'],
                    recurrent_dropout=params['recurrent_dropout'],
                    activation='relu')
    )
    model.add(
        layers.Dense(params['layer_size2'], activation='relu')
    )
    model.add(
        layers.Dense(params['layer_size3'], activation='relu')
    )
    model.add(
        layers.Dropout(params['second_dropout'])
    )
    model.add(
        layers.Dropout(0.3)
    )
    model.add(
        layers.Dense(params['layer_size4'], activation='relu')
    )
    model.add(
        layers.Dropout(0.3)
    )
    model.add(
        layers.Dense(params['layer_size4'], activation='relu')
    )
    model.add(
        layers.Dense(3, activation='sigmoid')
    )
    opt = optimizers.Adam(lr=0.000025)
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
    plt.savefig('./plots/'+str(len(os.listdir('./plots')) + 1)+'.png')


def train_nn(train_d, train_l, ep, params, tboard=False):
    m = RNN_model(params)
    print(m.summary())
    callbacks = [TerminateOnNaN()]
    if tboard == True:
        print("TENSORBOARD")
        tensorboard = TensorBoard(log_dir="./logs/{}".format(time()), histogram_freq=5, batch_size=64, write_images=True)
        callbacks.append(tensorboard)
    history = m.fit(train_d, train_l, epochs=ep, batch_size=int(params['batch_size']), validation_split=0.2
                    , callbacks=callbacks)
    return m, history


def eval_nn(model, test_d, test_l):
    return model.evaluate(test_d, test_l)


if __name__ == '__main__':
    test_data = np.load('./NP_Arrays/RNN/test_dataPTK_10.npy')
    test_label = np.load('./NP_Arrays/RNN/test_labelPTK_10.npy')
    train_data = np.load('./NP_Arrays/RNN/train_dataPTK_10.npy')
    train_label = np.load('./NP_Arrays/RNN/train_labelPTK_10.npy')
    params = {
        'epochs': 50,
        'batch_size': 64,
        'RNN_SIZE1': 960,
        'RNN_SIZE2': 400,
        'layer_size1': 300,
        'layer_size2': 150,
        'layer_size3': 80,
        'layer_size4': 40,
        'dropout': 0.5,
        'recurrent_dropout': 0.5,
        'second_dropout': 0.3
    }
    # mischio train e test set
    #total_data = np.concatenate((train_data, test_data))
    #total_label = np.concatenate((train_label, test_label))
    #index = np.random.permutation(77376)
    #train_data = total_data[index[0:61901]]
    #train_label = total_label[index[0:61901]]
    #test_data = total_data[index[61901:-1]]
    #test_label = total_label[index[61901:-1]]
    ################################################
    fit_result = train_nn(train_data, train_label, params['epochs'], params, tboard=True)
    test_result = eval_nn(fit_result[0], test_data, test_label)
    print(test_result)
    plot_value(params['epochs'], fit_result[1], test_result)
