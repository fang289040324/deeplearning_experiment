#!/usr/bin/env python
# coding: utf-8

import keras as k
import numpy as np
import tensorflow as tf
from sklearn import preprocessing as pre

batch_size = 800
epochs = 100000
validation_split = 0.1
lr = 1e-2
seq_size = 100  # 50ms * 100 = 5s
cell_size = 64
loss = k.losses.mse

# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def build_model(isBidirectional, isBatchNorm, isDropout):

    def bidirectionalCell(cell, isBidirectional):
        name = cell.name + '_bidi'
        if isBidirectional:
            return k.layers.Bidirectional(cell, name=name)
        else:
            return cell

    def batchNorm(layer, isBatchNorm):
        if isBatchNorm:
            return k.layers.BatchNormalization()(layer)
        else:
            return layer

    def dropout(layer, isDropout):
        if isDropout:
            return k.layers.GaussianDropout(0.3)(layer)
        else:
            return layer

    def conv_layer(layer, filter, kernel_size, strides_pool=(2, 2), pool_size=(2, 2), padding='same'):
        conv = k.layers.Conv2D(filter, kernel_size, padding=padding, activation='elu')(layer)
        return k.layers.MaxPool2D(pool_size=pool_size, strides=strides_pool, padding=padding)(conv)

    # def conv_layer(layer, filter):
    #     conv1x1_1 = k.layers.Conv2D(filter, (1, 1), activation='elu', padding='same')(layer)
    #     conv3x3_1 = k.layers.Conv2D(filter, (3, 3), activation='elu', padding='same')(conv1x1_1)
    #     conv1x3_1 = k.layers.Conv2D(filter, (1, 3), activation='elu', padding='same')(conv3x3_1)
    #     conv3x1_1 = k.layers.Conv2D(filter, (3, 1), activation='elu', padding='same')(conv3x3_1)
    #     conv1x1_2 = k.layers.Conv2D(filter, (1, 1), activation='elu', padding='same')(layer)
    #     conv1x3_2 = k.layers.Conv2D(filter, (1, 3), activation='elu', padding='same')(conv1x1_2)
    #     conv3x1_2 = k.layers.Conv2D(filter, (3, 1), activation='elu', padding='same')(conv1x1_2)
    #     conv1x1_3 = k.layers.Conv2D(filter, (1, 1), activation='elu', padding='same')(layer)
    #     pool_1 = k.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(layer)
    #     conv1x1_4 = k.layers.Conv2D(filter, (1, 1), activation='elu', padding='same')(pool_1)
    #     return k.layers.concatenate([conv1x3_1, conv3x1_1, conv1x3_2, conv3x1_2, conv1x1_3, conv1x1_4], axis=3)

    input_layer = k.layers.Input(shape=(100, 39, 1))
    gauss_input_layer = k.layers.GaussianNoise(0.1)(input_layer)

    conv = dropout(batchNorm(conv_layer(gauss_input_layer, 64, (5, 5)), isBatchNorm), isDropout)
    conv = dropout(batchNorm(conv_layer(conv, 128, (3, 3)), isBatchNorm), isDropout)
    conv = dropout(batchNorm(conv_layer(conv, 196, (3, 3)), isBatchNorm), isDropout)
    conv = dropout(batchNorm(conv_layer(conv, 256, (2, 2)), isBatchNorm), isDropout)
    conv = dropout(batchNorm(conv_layer(conv, 512, (2, 2), padding='valid'), isBatchNorm), isDropout)
    reshape = k.layers.Reshape((1, -1, 1))(conv)
    conv = dropout(batchNorm(conv_layer(reshape, 20, (1, 1), (1, 1), (1, 1)), isBatchNorm), isDropout)
    conv = dropout(batchNorm(conv_layer(conv, 20, (1, 1), (1, 1), (1, 1)), isBatchNorm), isDropout)
    conv = dropout(batchNorm(conv_layer(conv, 20, (1, 1), (1, 1), (1, 1)), isBatchNorm), isDropout)
    reshape = k.layers.Reshape((-1, 20))(conv)
    permute = k.layers.Permute((2, 1))(reshape)

    cell = dropout(batchNorm(bidirectionalCell(k.layers.LSTM(cell_size, return_sequences=True, kernel_regularizer=k.regularizers.l2(0.001),
                                                             activity_regularizer=k.regularizers.l2(0.001)), isBidirectional)(permute), isBatchNorm), isDropout)
    cell = dropout(batchNorm(bidirectionalCell(k.layers.LSTM(cell_size, return_sequences=True, kernel_regularizer=k.regularizers.l2(0.001),
                                                             activity_regularizer=k.regularizers.l2(0.001)), isBidirectional)(cell), isBatchNorm), isDropout)
    cell = batchNorm(k.layers.LSTM(1, return_sequences=True, kernel_regularizer=k.regularizers.l2(0.001))(cell), isBatchNorm)

    model = k.models.Model(input_layer, cell)
    model.summary()
    model.compile(optimizer=k.optimizers.Adam(lr=lr), loss=loss)
    return model


def train():
    model = build_model(True, True, True)
    try:
        model.load_weights('model/bmw/checkpoint/cnn_rnn-1199-val_loss_0.35-loss_0.04.hdf5')
        print('model load complete !!')
    except Exception:
        print('not model!')
    model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs,
              callbacks=[k.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=10, verbose=1, cooldown=50),
                  k.callbacks.ModelCheckpoint(filepath='./model/bmw/checkpoint/cnn_rnn_l2-{epoch:02d}-val_loss_{val_loss:.2f}-loss_{loss:.2f}.hdf5', verbose=1, period=600, save_best_only=False, mode='min'),
                  k.callbacks.TensorBoard(histogram_freq=50, write_graph=True, write_images=False)],
              validation_split=validation_split, shuffle=False)


if __name__ == '__main__':
    # x_main = np.load('./data/bmw/x_main_all.npy').astype(np.float64)
    # x_main = x_main.reshape([-1, 24])
    # x_main = pre.scale(x_main)
    #
    # x_aux = np.load('./data/bmw/x_aux_all.npy').astype(np.float64)
    # x_aux = x_aux.reshape([-1, 15])
    # x_aux = pre.scale(x_aux)
    #
    # print(x_main.shape)
    # print(x_aux.shape)
    #
    # x = np.concatenate((x_main, x_aux), axis=1)
    # x = x[:-80].reshape([-1, seq_size, 39, 1])
    # print(x.shape)
    #
    # y = np.load('./data/bmw/y_all.npy').astype(np.float64)
    # y = y.reshape([-1, 1])
    # y_result = []
    # print(y.shape[0])
    # print(int(y.shape[0] / seq_size))
    # for i in range(int(y.shape[0] / seq_size)):
    #     i += 1
    #     for d in y[(i * seq_size): (i * seq_size + 20)]:
    #         y_result.append(d)
    #
    # print(len(y_result))
    # y = np.array(y_result).reshape([-1, 20, 1])
    # print(y.shape)

    x_main = np.load('./data/bmw/x_main_all.npy').astype(np.float64)
    x_main = x_main.reshape([-1, 24])
    x_main = pre.scale(x_main)

    x_aux = np.load('./data/bmw/x_aux_all.npy').astype(np.float64)
    x_aux = x_aux.reshape([-1, 15])
    x_aux = pre.scale(x_aux)

    print(x_main.shape)
    print(x_aux.shape)

    x = np.concatenate((x_main, x_aux), axis=1)
    print(x.shape)

    x_ = []
    for i in range(int(x.shape[0] / 20)):
        if x[i * 20: i * 20 + seq_size].shape[0] == 100:
            x_.append(x[i * 20: i * 20 + seq_size])

    x_ = np.array(x_)
    x = x_.reshape(-1, 100, 39, 1)[:-1]
    print(x.shape)

    y = np.load('./data/bmw/y_all.npy').astype(np.float64)
    y = y.reshape([-1, 1])[100:].reshape([-1, 20, 1])
    print(y.shape)

    train()
