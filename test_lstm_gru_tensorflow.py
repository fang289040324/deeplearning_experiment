#!/usr/bin/env python
# coding=utf-8

import tensorflow.contrib as tc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import time
from sklearn import preprocessing as pre

# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

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
lr = 1e-2
epochs = 100000


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
    cell = tc.rnn.AttentionCellWrapper(
        tc.rnn.ResidualWrapper(
            tc.rnn.MultiRNNCell(
                [tc.rnn.LayerNormBasicLSTMCell(dec_units) for _ in range(dec_layers)] +
                [tc.rnn.LayerNormBasicLSTMCell(1)])),
        attn_length=attn_len, state_is_tuple=True)
    state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    output, state = tf.nn.dynamic_rnn(cell, x, initial_state=state)
    return output


def build_all_bidi(x):
    attn_len = 10
    dec_layers = 2
    dec_units = 256

    x = tf.unstack(x, 10, 1)

    cell1 = tc.rnn.AttentionCellWrapper(
        tc.rnn.ResidualWrapper(
            tc.rnn.MultiRNNCell(
                [tc.rnn.LayerNormBasicLSTMCell(dec_units) for _ in range(dec_layers)] +
                [tc.rnn.LayerNormBasicLSTMCell(1)])),
        attn_length=attn_len, state_is_tuple=True)
    state1 = cell1.zero_state(batch_size=batch_size, dtype=tf.float32)
    cell2 = tc.rnn.AttentionCellWrapper(
        tc.rnn.ResidualWrapper(
            tc.rnn.MultiRNNCell(
                [tc.rnn.LayerNormBasicLSTMCell(dec_units) for _ in range(dec_layers)] +
                [tc.rnn.LayerNormBasicLSTMCell(1)])),
        attn_length=attn_len, state_is_tuple=True)
    state2 = cell2.zero_state(batch_size=batch_size, dtype=tf.float32)
    (output, _, _) = tc.rnn.static_bidirectional_rnn(cell1, cell2, x, state1, state2)
    return output


def seq2seq(x):
    n_hidden = 32

    # Encoder LSTM cells
    lstm_fw_cell = tc.rnn.BasicLSTMCell(n_hidden)
    lstm_bw_cell = tc.rnn.BasicLSTMCell(n_hidden)

    state_fw = lstm_fw_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    state_bw = lstm_bw_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

    # Bidirectional RNN
    (encoder_outputs, _, _) = tc.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, state_fw, state_bw, dtype=tf.float32)

    # Decoder LSTM cell
    decoder_cell = tc.rnn.BasicLSTMCell(n_hidden)

    # Attention mechanism
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(n_hidden, encoder_outputs)
    # attn_cell 是隐层的输出,这个值需要传到下一个时间状态的 decoder, 并非是 decoder 的输出
    attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_size=n_hidden)

    # Initial attention
    attn_zero = attn_cell.zero_state(batch_size=tf.shape(x)[0], dtype=tf.float32)

    # Helper function
    helper = tc.seq2seq.TrainingHelper(inputs=None)

    # Decoding
    my_decoder = tc.seq2seq.BasicDecoder(cell=attn_cell, helper=helper, initial_state=attn_zero)

    decoder_outputs, decoder_states = tc.seq2seq.dynamic_decode(my_decoder)


def train():
    input_tensor = tf.placeholder(tf.float32, (None, 10, 1))
    y = tf.placeholder(tf.float32, [None, 10, 1])

    # y_ = build_nas(input_tensor)
    # y_ = build_lstm(input_tensor)
    # y_ = build_gru(input_tensor)
    # y_ = build_all(input_tensor)
    y_ = build_all_bidi(input_tensor)
    y_ = tf.stack(y_, axis=1)
    print(y_[0].shape)

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
