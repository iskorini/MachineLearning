from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


def RNN_model(params):
    model = models.Sequential()
    model.add(
        layers.LSTM(32, input_shape=(1, 120))
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


def train_nn(train_d, train_l, ep, params):
    m = RNN_model(params)
    print(m.summary())
    history = m.fit(train_d, train_l, epochs=ep, batch_size=int(params['batch_size']), validation_split=0.2)
    return m, history


def eval_nn(model, test_d, test_l):
    return model.evaluate(test_d, test_l)


if __name__ == '__main__':
    test_data = np.load("./NP_Arrays/RNN/test_dataPTK.npy")
    test_label = np.load("./NP_Arrays/RNN/test_labelPTK.npy")
    train_data = np.load("./NP_Arrays/RNN/train_dataPTK.npy")
    train_label = np.load("./NP_Arrays/RNN/train_labelPTK.npy")
    test_data = test_data.reshape((test_data.shape[0], 1, 120))
    train_data = train_data.reshape((train_data.shape[0], 1, 120))
    epochs = 100
    params = {
        'batch_size': 4,
        'layer_size1': 1000,
        'layer_size2': 120
    }
    fit_result = train_nn(train_data, train_label, epochs, params)
    test_result = eval_nn(fit_result[0], test_data, test_label)
    plot_value(epochs, fit_result[1], test_result)
    exit(0)
