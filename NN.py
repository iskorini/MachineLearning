import os
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


def RNN_model(params):
    model = models.Sequential()
    model.add(
        layers.Bidirectional(layers.LSTM(params['LSTM_SIZE']), input_shape=(10, 120))
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
    plt.savefig('./plots/'+str(len(os.listdir('./plots')) + 1)+'.png')


def train_nn(train_d, train_l, ep, params):
    m = RNN_model(params)
    print(m.summary())
    history = m.fit(train_d, train_l, epochs=ep, batch_size=int(params['batch_size']), validation_split=0.2)
    return m, history


def eval_nn(model, test_d, test_l):
    return model.evaluate(test_d, test_l)


if __name__ == '__main__':
    test_data = np.load('./NP_Arrays/RNN/test_dataPTK_10.npy')
    test_label = np.load('./NP_Arrays/RNN/test_labelPTK_10.npy')
    train_data = np.load('./NP_Arrays/RNN/train_dataPTK_10.npy')
    train_label = np.load('./NP_Arrays/RNN/train_labelPTK_10.npy')
    params = {
        'epochs': 100,
        'batch_size': 4,
        'LSTM_SIZE': 32,
        'layer_size2': 120
    }
    fit_result = train_nn(train_data, train_label, params['epochs'], params)
    test_result = eval_nn(fit_result[0], test_data, test_label)
    plot_value(params['epochs'], fit_result[1], test_result)
    exit(0)
