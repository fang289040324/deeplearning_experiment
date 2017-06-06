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

model = k.models.load_model('model/bmw/checkpoint/model_128_noise_all3-1499-0.16-0.47.hdf5')
# model = k.models.load_model('model/bmw/checkpoint/model_128_noise_all2-1799-0.16-0.48.hdf5')
# model = k.models.load_model('model/bmw/checkpoint/model_128_noise_all-7199-0.19-0.46.hdf5')
# model = k.models.load_model('model/bmw/checkpoint/model_64_noise_all-599-0.17-0.34.hdf5')
# model = k.models.load_model('model-3499-0.28.hdf5')
# model = k.models.load_model('model_128-499-0.28.hdf5')

x_aux = np.load('data/bmw/x_aux_all.npy')
x_aux = pre.scale(x_aux.reshape([-1, 60 * 15]))
x_aux = x_aux.reshape([-1, 20, 15])
x_main = np.load('data/bmw/x_main_all.npy')
x_main = pre.scale(x_main.reshape([-1, 60 * 24]))
x_main = x_main.reshape([-1, 20, 24])
y = np.load('data/bmw/y_all.npy').reshape([-1, 60]).reshape([-1, 20, 1])

# x_aux = np.load('data/bmw/x_aux_all.npy')
# x_aux = pre.scale(x_aux.reshape([-1, 60 * 15]))
# x_aux = x_aux.reshape([-1, 20, 15])
# x_main = np.load('data/bmw/x_main_all.npy')
# x_main = pre.scale(x_main.reshape([-1, 60 * 24]))
# x_main = x_main.reshape([-1, 20, 24])
# y = np.load('data/bmw/y_all.npy').reshape([-1, 60]).reshape([-1, 20, 1])

y_1, y_2, y_3 = model.predict([x_main, x_aux], batch_size=500, verbose=1)

plot.plot(y.reshape([-1]), 'r', label='y')
plot.plot((y_1 * 0.7 + y_2 * 0.1 + y_3 * 0.2).reshape([-1]), 'g', label='y_')
plot.legend()
plot.show()
