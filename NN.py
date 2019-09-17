import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto() 
# dynamically grow GPU memory 
#config.gpu_options.allow_growth = True 
set_session(tf.Session(config=config))
import wandb
from wandb.keras import WandbCallback
from time import time
from keras import layers, models, optimizers, regularizers
from keras.callbacks import TensorBoard, TerminateOnNaN, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from dl_bot import DLBot
from telegram_bot_callback import TelegramBotCallback


rcParams.update({'figure.autolayout': True})

def plot_value(history, ev):
    fig, ax = plt.subplots()
    ep = len(history.history['val_loss'])
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
    file_name = './plots2/' + str(len(os.listdir('./plots2')) + 1) + '_RNN.eps'
    file_name = './plots2/W11_RNN.eps'
    print(file_name)
    plt.savefig(file_name, format = 'eps')


def RNN_model_DEFINITIVE(params, multi_gpu):

    model = models.Sequential()
    model.add(
        layers.Bidirectional(layers.LSTM(800, activation='relu', return_sequences= True, recurrent_dropout = 0.2), input_shape=(1,123))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(200, activation='relu', return_sequences= False, dropout = 0.2, recurrent_dropout = 0.2))
    )
    model.add(
        layers.Dropout(0.3)
    )
    model.add(
        layers.Dense(500, activation = 'relu')
    )
    model.add(
        layers.Dropout(0.3)
    )
    model.add(
        layers.Dense(500, activation = 'relu')
    )
    model.add(
        layers.Dense(40, activation='softmax')
    )
    opt = optimizers.Adam() 
    if multi_gpu:
        print("--------------- MULTIPLE GPU ---------------")
        model = multi_gpu_model(model, gpus=2)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['acc']
    )
    return model



def simple_model_DEFINITIVE(params):
    model = models.Sequential()
    model.add(
        layers.Dense(123, activation = 'relu', input_shape = (123,))
    )
    model.add(
        layers.Dropout(0.2)
    )
    model.add(
        layers.Dense(2000, activation = 'relu')
    )
    model.add(
        layers.Dropout(0.2)
    )
    model.add(
        layers.Dense(1000, activation='relu')
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



def RNN_model(params, multi_gpu):
    model = models.Sequential()
    model.add(
        layers.Bidirectional(layers.LSTM(600, activation='relu', return_sequences= True, recurrent_dropout = 0.3)
        , input_shape=(11, 123))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(200, activation='relu', dropout = 0.3, recurrent_dropout = 0.3))
    )
    model.add(
        layers.Dropout(0.3)
    )
    model.add(
        layers.BatchNormalization()
    )
    model.add(
        layers.Dense(650, activation = 'relu')
    )
    model.add(
        layers.Dropout(0.3)
    )
    model.add(
        layers.Dense(650, activation = 'relu')
    )
    model.add(
        layers.Dense(40, activation='softmax')
    )
    opt = optimizers.Adam(lr = 0.0005) #half the std learning rate
    if multi_gpu:
        print("--------------- MULTIPLE GPU ---------------")
        model = multi_gpu_model(model, gpus=2)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['acc']
    )
    return model


def RNN_model2(params, multi_gpu):
    model = models.Sequential()
    model.add(
        layers.Bidirectional(layers.SimpleRNN(700, activation='relu', return_sequences= True, recurrent_dropout = 0.3)
        , input_shape=(11, 123))
    )
    model.add(
        layers.Bidirectional(layers.SimpleRNN(200, activation='relu', dropout = 0.3, recurrent_dropout = 0.3))
    )
    model.add(
        layers.Dropout(0.3)
    )
    model.add(
        layers.BatchNormalization()
    )
    model.add(
        layers.Dense(700, activation = 'relu')
    )
    model.add(
        layers.Dropout(0.3)
    )
    model.add(
        layers.Dense(700, activation = 'relu')
    )
    model.add(
        layers.Dense(40, activation='softmax')
    )
    opt = optimizers.Adam(lr = 0.0005) #half the std learning rate
    if multi_gpu:
        print("--------------- MULTIPLE GPU ---------------")
        model = multi_gpu_model(model, gpus=2)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['acc']
    )
    return model


def train_nn(train_d, train_l, val,  ep, params, tboard=False, multi_gpu = False):
    m = RNN_model2(params, multi_gpu)
    print(m.summary())
    callbacks = [TerminateOnNaN(), CSVLogger('./csv_log/' + str(len(os.listdir('./plots')) + 1) + '_CSV.csv')]
    if tboard == True:
        print("TENSORBOARD")
        tensorboard = TensorBoard(log_dir="./logs/{}".format(time()), histogram_freq=5, 
            batch_size=int(params['batch_size']), write_images=True)
        callbacks.append(tensorboard)
    #######################################
    telegram_token = "****"
    telegram_user_id = 25897312
    bot = DLBot(token=telegram_token, user_id=telegram_user_id)
    callbacks.append(TelegramBotCallback(bot))
    #######################################
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=5, min_delta=0.01)
    #callbacks.append(es)
    #######################################
    mc = ModelCheckpoint('./checkpoint/' + str(len(os.listdir('./plots')) + 1) + '_LSTM.h5', monitor='val_acc', verbose=1, save_best_only=True)
    #callbacks.append(mc)
    #######################################
    callbacks.append(WandbCallback())
    #######################################
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
    wandb.init()
    test_data = np.load('./NP_Arrays/RNN/LIBROSA/W11_40_01/test_data.npy')
    test_label = np.load('./NP_Arrays/RNN/LIBROSA/W11_40_01/test_label.npy')
    train_data = np.load('./NP_Arrays/RNN/LIBROSA/W11_40_01/train_data.npy')
    train_label = np.load('./NP_Arrays/RNN/LIBROSA/W11_40_01/train_label.npy')
    validation_data = np.load('./NP_Arrays/RNN/LIBROSA/W11_40_01/validation_data.npy')
    validation_label = np.load('./NP_Arrays/RNN/LIBROSA/W11_40_01/validation_label.npy')
    params = {
        'epochs': 150,
        'batch_size': 500, #337,
        'lr': 0.0005
    }
    print(params)
    print(train_data.shape, train_label.shape)
    print(test_data.shape, test_label.shape)
    print(validation_data.shape, validation_label.shape)
    fit_result = train_nn(train_data, train_label, [validation_data, validation_label],params['epochs'], params, tboard=False)
    test_result = eval_nn(fit_result[0], test_data, test_label)
    print(test_result)
    plot_value(fit_result[1], test_result)



                                
                  
