#!/usr/bin/env python
# coding=utf-8

import keras as k
import numpy as np
import matplotlib.pyplot as plot

model = k.models.load_model('model-3499-0.28.hdf5')
# model = k.models.load_model('model_128-499-0.28.hdf5')
# x_aux = np.load('x_aux.npy')
# x_main = np.load('x_main.npy')
# y = np.load('y.npy')
x_aux = np.load('x_aux_all.npy')
x_main = np.load('x_main_all.npy')
y = np.load('y_all.npy')

y_, _ = model.predict([x_main, x_aux], batch_size=500, verbose=1)

y_ = y_.reshape([-1])
print(y_.shape)
for i in range(y_.size):
    print(i, i, '-', (i + 60), (y_[i: (i + 60)].max() - y_[i: (i + 60)].min()))

plot.plot(y.reshape((-1)), 'r')
plot.show()
plot.plot(y.reshape([-1]), 'r', label='y')
plot.plot(y_.reshape([-1]), 'g', label='y_')
plot.legend()
plot.show()
