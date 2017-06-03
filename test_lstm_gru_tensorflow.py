#!/usr/bin/env python
# coding=utf-8

import tensorflow.contrib as tc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import time
from sklearn import preprocessing as pre

x_data = np.linspace(0, 200, num=10000)
# x_data = pre.scale(x_data)
# x_data *= 10
y_data = np.cos(x_data) + 10
# y_data = pre.scale(y_data)
# y_data = np.cos(x_data)

print(x_data.shape, y_data.shape)
x_data = x_data.reshape([-1, 10, 1])
y_data = y_data.reshape([-1, 10, 1])
print(x_data[0: 999].shape, y_data[1:1000].shape)

batch_size = 999
lr = 1e-3
epochs = 1000


def build_gru(input_tensor):
    cell = tc.rnn.MultiRNNCell(
        [tc.rnn.ResidualWrapper(tc.rnn.GRUCell(64)) for _ in range(1)] + [tc.rnn.ResidualWrapper(tc.rnn.GRUCell(1))])
    cell_init = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    output, state = tf.nn.dynamic_rnn(cell, inputs=input_tensor, initial_state=cell_init)
    print('output:', output)
    print('state:', state)
    return output


def build_lstm(x):
    rnn_cell = tc.rnn.MultiRNNCell([tc.rnn.LSTMCell(64) for _ in range(1)] + [tc.rnn.LSTMCell(1)])
    state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    output, state = tf.nn.dynamic_rnn(rnn_cell, inputs=x, initial_state=state)
    return output


def build_nas(x):
    rnn_cell = tc.rnn.MultiRNNCell([tc.rnn.NASCell(64) for _ in range(1)] + [tc.rnn.NASCell(1)])
    state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    output, state = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=state)
    return output


def build_all(x):
    attn_len = 5
    dec_layers = 2
    dec_units = 32
    cell = tc.rnn.AttentionCellWrapper(tc.rnn.MultiRNNCell([tc.rnn.ResidualWrapper(cell=tc.rnn.LSTMCell(dec_units)) for _ in range(dec_layers)]), attn_length=attn_len)
    cell.zero_state()


def train():
    input_tensor = tf.placeholder(tf.float32, (None, 10, 1))
    y = tf.placeholder(tf.float32, [None, 10, 1])

    # y_ = build_nas(input_tensor)
    # y_ = build_lstm(input_tensor)
    y_ = build_gru(input_tensor)

    # loss = -tf.reduce_mean(y * tf.log(y_))
    loss = tf.reduce_mean(tc.keras.losses.mean_absolute_error(y, y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    accuracy = tf.reduce_mean(tc.keras.metrics.mean_absolute_error(y, y_))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for i in range(epochs):
            start_time_epoch = time.time()
            sess.run(optimizer, feed_dict={input_tensor: x_data[0:999], y: y_data[1:1000]})
            end_time_epoch = time.time()

            print('epoch:', i, 'loss time:', round((end_time_epoch - start_time_epoch), 3), 'loss:',
                  sess.run(loss, feed_dict={input_tensor: x_data[0:999], y: y_data[1:1000]}),
                  'accuracy:', sess.run(accuracy, feed_dict={input_tensor: x_data[0:999], y: y_data[1:1000]}))
        end_time = time.time()
        print('cost time:', (end_time - start_time))


if __name__ == '__main__':
    train()
