#!/usr/bin/env python
# coding=utf-8

import keras as k
import numpy as np
import matplotlib.pyplot as plot
from sklearn import preprocessing as pre
import tensorflow as tf


# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

seq_size = 100

model = k.models.load_model('model/bmw/checkpoint/cnn_rnn11_l2-3599-val_loss_0.28-loss_0.02.hdf5')

x_main = np.load('./data/bmw/x_main_all.npy').astype(np.float64)
x_main = x_main.reshape([-1, 24])
x_main = pre.scale(x_main)

x_aux = np.load('./data/bmw/x_aux_all.npy').astype(np.float64)
x_aux = x_aux.reshape([-1, 15])
x_aux = pre.scale(x_aux)

print(x_main.shape)
print(x_aux.shape)

x = np.concatenate((x_main, x_aux), axis=1)
# x = x[:-40].reshape([-1, seq_size, 39, 1])
print(x.shape)
print(x.shape[0])
# x_ = np.ones([seq_size, 39])
# for i in range(int(x.shape[0] / 20)):
#     if i == 0:
#         x_ *= x[i * 20: i * 20 + seq_size]
#     elif x[i * 20: i * 20 + seq_size].shape[0] == 100:
#         x_ = np.concatenate((x_, x[i * 20: i * 20 + seq_size]))
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


y_ = model.predict(x, batch_size=500, verbose=1)

plot.plot(y.reshape([-1]), 'r', label='y')
plot.plot(y_.reshape([-1]), 'g', label='y_')
plot.legend()
plot.show()
