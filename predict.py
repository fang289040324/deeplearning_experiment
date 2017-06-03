#!/usr/bin/env python
# coding=utf-8

import keras as k
import numpy as np
import matplotlib.pyplot as plot

model = k.models.load_model('model/bmw/checkpoint/model_32_noise-1514-0.25.hdf5')
# model = k.models.load_model('model-3499-0.28.hdf5')
# model = k.models.load_model('model_128-499-0.28.hdf5')

x_aux = np.load('data/bmw/x_aux.npy')
x_main = np.load('data/bmw/x_main.npy')
y = np.load('data/bmw/y.npy')

# x_aux = np.load('x_aux_all.npy')
# x_main = np.load('x_main_all.npy')
# y = np.load('y_all.npy')

y_1, y_2, y_3 = model.predict([x_main, x_aux], batch_size=500, verbose=1)

plot.plot(y.reshape([-1]), 'r', label='y')
plot.plot((y_1 * 0 + y_2 * 0 + y_3 * 1).reshape([-1]), 'g', label='y_')
plot.legend()
plot.show()
