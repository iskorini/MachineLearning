import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  
from time import time
from keras import layers, models, optimizers
from keras.callbacks import TensorBoard, TerminateOnNaN
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
rcParams.update({'figure.autolayout': True})

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
    print('./plots/' + str(len(os.listdir('./plots')) + 1) + 'DENSE2.png')
    plt.savefig('./plots/' + str(len(os.listdir('./plots')) + 1) + 'DENSE2.png')


def RNN_model(params):
    model = models.Sequential()
    model.add(
            layers.Bidirectional(
                layers.LSTM(123, activation='relu', return_sequences = True),  input_shape=(1, 123)
                    )
    )
    model.add(
        layers.BatchNormalization()
    )
    model.add(
            layers.Bidirectional(
                layers.LSTM(200, activation='relu'))
    )
    model.add(
        layers.Dropout(0.3)
    )
    model.add(
        layers.Dense(300, activation='relu')
    )
    model.add(
        layers.Dense(61, activation='softmax')
    )
    opt = optimizers.SGD(lr = 1/1000, momentum = 0.9)
    #opt = optimizers.SGD(lr = 0.00001, momentum = 0.9)  #default 1/1000
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['acc']
    )
    return model


def simple_model(params):
    model = models.Sequential()
    model.add(
        layers.Dense(123, activation = 'relu', input_shape = (123*3,))
    )
    model.add(
        layers.Dropout(0.4)
    )
    model.add(
        layers.Dense(1000, activation = 'relu')
    )
    model.add(
        layers.Dense(40, activation='softmax')
    )
    opt = optimizers.Adam()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['acc']
    )
    return model

def RNN_model2(params):
    model = models.Sequential()
    model.add(
        layers.LSTM(123, activation='relu', dropout = 0.4, recurrent_dropout = 0.4, return_sequences = True, input_shape=(3, 123))
    )
    model.add(
        layers.LSTM(200, activation='relu')
    )
    model.add(
        layers.Dropout(0.3)
    )
    model.add(
        layers.Dense(300, activation='relu')
    )
    model.add(
        layers.Dense(40, activation='softmax')
    )
    opt = optimizers.Adam()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['acc']
    )
    return model


def train_nn(train_d, train_l, val,  ep, params, tboard=False):
    #m = RNN_model2(params)
    m = simple_model(params)
    #m = test_model()
    print(m.summary())
    callbacks = [TerminateOnNaN()]
    if tboard == True:
        print("TENSORBOARD")
        tensorboard = TensorBoard(log_dir="./logs/{}".format(time()), histogram_freq=5, batch_size=64, write_images=True)
        callbacks.append(tensorboard)
    history = m.fit(train_d, train_l, epochs=ep, batch_size=int(params['batch_size']),validation_data = val
                    , callbacks=callbacks)
    return m, history


def eval_nn(model, test_d, test_l):
    return model.evaluate(test_d, test_l)


def encode_label(train, test, val):
    encoder = LabelEncoder()
    encoder.fit(
        np.concatenate(
            (train, np.concatenate((test, val)))
            )
        )
    train_encoded_labels = encoder.transform(train)
    test_encoded_labels = encoder.transform(test)
    val_encoded_labels = encoder.transform(val)
    return to_categorical(train_encoded_labels), to_categorical(test_encoded_labels), to_categorical(val_encoded_labels)


if __name__ == '__main__':
    test_data = np.load('./NP_Arrays/RNN/LIBROSA/test_data.npy')
    test_label = np.load('./NP_Arrays/RNN/LIBROSA/test_label.npy')
    train_data = np.load('./NP_Arrays/RNN/LIBROSA/train_data.npy')
    train_label = np.load('./NP_Arrays/RNN/LIBROSA/train_label.npy')
    validation_data = np.load('./NP_Arrays/RNN/LIBROSA/validation_data.npy')
    validation_label = np.load('./NP_Arrays/RNN/LIBROSA/validation_label.npy')
    #test_data = test_data.reshape(test_data.shape[0], 1, 123)
    #train_data = train_data.reshape(train_data.shape[0], 1, 123)
    #validation_data = validation_data.reshape(validation_data.shape[0], 1, 123)
    #train_label, test_label, validation_label = encode_label(train_label, test_label, validation_label)
    params = {
        'epochs': 5,
        'batch_size': 240,
    }
    print(params)
    print(train_data.shape, train_label.shape)
    print(test_data.shape, test_label.shape)
    print(validation_data.shape, validation_label.shape)
    # mischio train e test set
    #total_data = np.concatenate((train_data, test_data))
    #total_label = np.concatenate((train_label, test_label))
    #index = np.random.permutation(77376)
    #train_data = total_data[index[0:61901]]
    #train_label = total_label[index[0:61901]]
    #test_data = total_data[index[61901:-1]]
    #test_label = total_label[index[61901:-1]]
    ################################################
    fit_result = train_nn(train_data, train_label, [validation_data, validation_label],params['epochs'], params, tboard=False)
    test_result = eval_nn(fit_result[0], test_data, test_label)
    print(test_result)
    plot_value(params['epochs'], fit_result[1], test_result)



                                
                  