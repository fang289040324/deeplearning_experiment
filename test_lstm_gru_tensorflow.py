#!/usr/bin/env python
# coding=utf-8

import tensorflow.contrib as tc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot

x_data = np.linspace(0, 200, num=10000)
# x_data *= 10
y_data = np.cos(x_data) + 10
# y_data = np.cos(x_data)

print(x_data.shape, y_data.shape)
print(x_data[0:10], y_data[10:20])
x_data = x_data.reshape([-1, 10, 1])
y_data = y_data.reshape([-1, 10, 1])
print(x_data[0:99].shape, y_data[1:100].shape)
print(x_data[0], y_data[1])

batch_size = 3000
lr = 1e-3


def build_gru():
    input_tensor = tf.placeholder(tf.float32, (10, 1))
    gru_cell = tc.rnn.GRUCell(256)
    pass


def build_lstm():
    pass


def train():
    pass


plot.plot(y_data.reshape([-1]), 'r', label='y')
plot.plot(y_.reshape([-1]), 'g', label='y_')
plot.legend()
plot.show()
