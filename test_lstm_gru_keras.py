#!/usr/bin/env python
# coding=utf-8

import keras as k
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plot

x_data = np.linspace(0, 200, num=10000)
# x_data *= 10
y_data = np.cos(x_data) + 10
# y_data = np.cos(x_data)

print x_data.shape, y_data.shape
print x_data[0:10], y_data[10:20]
x_data = x_data.reshape([-1, 10, 1])
y_data = y_data.reshape([-1, 10, 1])
print x_data[0:99].shape, y_data[1:100].shape
print x_data[0], y_data[1]

batch_size = 3000
lr = 1e-3


def build_gru():
    input_layer = k.layers.Input((10, 1))
    gru_cell = k.layers.Bidirectional(k.layers.GRU(256, return_sequences=True))(input_layer)
    batchNorm = k.layers.BatchNormalization()(gru_cell)

    # gru_cell = k.layers.Bidirectional(k.layers.LSTM(256, return_sequences=True))(gru_cell)
    # gru_cell = k.layers.Bidirectional(k.layers.LSTM(256, return_sequences=True))(gru_cell)
    # output_layer = k.layers.LSTM(1, return_sequences=True)(gru_cell)

    flatten = k.layers.Flatten()(batchNorm)
    encode_output = k.layers.Dense(512)(flatten)
    batchNorm = k.layers.BatchNormalization()(encode_output)
    encode_output = k.layers.RepeatVector(10)(batchNorm)
    batchNorm = k.layers.BatchNormalization()(encode_output)

    gru_cell = k.layers.Bidirectional(k.layers.GRU(256, return_sequences=True))(batchNorm)
    batchNorm = k.layers.BatchNormalization()(gru_cell)
    output_layer = k.layers.TimeDistributed(k.layers.Dense(1))(batchNorm)

    model = k.models.Model(input_layer, output_layer)
    model.summary()

    return model


def build_lstm():
    input_layer = k.layers.Input((10, 1))
    gru_cell = k.layers.Bidirectional(k.layers.LSTM(256, return_sequences=True))(input_layer)
    batchNorm = k.layers.BatchNormalization()(gru_cell)

    # gru_cell = k.layers.Bidirectional(k.layers.LSTM(256, return_sequences=True))(gru_cell)
    # gru_cell = k.layers.Bidirectional(k.layers.LSTM(256, return_sequences=True))(gru_cell)
    # output_layer = k.layers.LSTM(1, return_sequences=True)(gru_cell)

    flatten = k.layers.Flatten()(batchNorm)
    encode_output = k.layers.Dense(512)(flatten)
    batchNorm = k.layers.BatchNormalization()(encode_output)
    encode_output = k.layers.RepeatVector(10)(batchNorm)
    batchNorm = k.layers.BatchNormalization()(encode_output)

    gru_cell = k.layers.Bidirectional(k.layers.LSTM(256, return_sequences=True))(batchNorm)
    batchNorm = k.layers.BatchNormalization()(gru_cell)
    output_layer = k.layers.TimeDistributed(k.layers.Dense(1))(batchNorm)

    model = k.models.Model(input_layer, output_layer)
    model.summary()

    return model


def build_GRU_attention():
    input_layer = k.layers.Input((10, 1))
    gru_cell = k.layers.Bidirectional(k.layers.GRU(256, return_sequences=True))(input_layer)
    batchNorm = k.layers.BatchNormalization()(gru_cell)

    # gru_cell = k.layers.Bidirectional(k.layers.LSTM(256, return_sequences=True))(gru_cell)
    # gru_cell = k.layers.Bidirectional(k.layers.LSTM(256, return_sequences=True))(gru_cell)
    # output_layer = k.layers.LSTM(1, return_sequences=True)(gru_cell)

    flatten = k.layers.Flatten()(batchNorm)
    encode_output = k.layers.Dense(512)(flatten)
    batchNorm = k.layers.BatchNormalization()(encode_output)
    encode_output = k.layers.RepeatVector(10)(batchNorm)
    batchNorm = k.layers.BatchNormalization()(encode_output)

    # def attention(query):
    #   """Put attention masks on hidden using hidden_features and query."""
    #   ds = []  # Results of attention reads will be stored here.
    #   if nest.is_sequence(query):  # If the query is a tuple, flatten it.
    #     query_list = nest.flatten(query)
    #     for q in query_list:  # Check that ndims == 2 if specified.
    #       ndims = q.get_shape().ndims
    #       if ndims:
    #         assert ndims == 2
    #     query = array_ops.concat(query_list, 1)
    #   for a in xrange(num_heads):
    #     with variable_scope.variable_scope("Attention_%d" % a):
    #       y = linear(query, attention_vec_size, True)
    #       y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
    #       # Attention mask is a softmax of v^T * tanh(...).
    #       s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
    #                               [2, 3])
    #       a = nn_ops.softmax(s)
    #       # Now calculate the attention-weighted vector d.
    #       d = math_ops.reduce_sum(
    #           array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
    #       ds.append(array_ops.reshape(d, [-1, attn_size]))
    #   return ds

    gru_cell = k.layers.Bidirectional(k.layers.GRU(256, return_sequences=True))(batchNorm)
    batchNorm = k.layers.BatchNormalization()(gru_cell)
    output_layer = k.layers.TimeDistributed(k.layers.Dense(1))(batchNorm)

    model = k.models.Model(input_layer, output_layer)
    model.summary()

    return model


def train():
    # gru_model = build_lstm()
    gru_model = build_gru()
    gru_model.compile(optimizer=k.optimizers.Nadam(lr=lr), loss=k.losses.mae, metrics=[k.metrics.mae])
    gru_model.fit(x=x_data[0:999], y=y_data[1:1000], batch_size=batch_size, epochs=1000, validation_split=0.1,
                  callbacks=[k.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=20, verbose=1)])
    return gru_model


model = train()

y_ = model.predict(x=x_data, verbose=1)

plot.plot(y_data.reshape([-1]), 'r', label='y')
plot.plot(y_.reshape([-1]), 'g', label='y_')
plot.legend()
plot.show()
