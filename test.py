import tensorflow.contrib as contrib
import tensorflow as tf

l = contrib.rnn.BasicLSTMCell(10)
rnn_cell = contrib.rnn.MultiRNNCell([l] * 5)
output, state = tf.nn.dynamic_rnn(rnn_cell, tf.placeholder(tf.float32, [None, None, 10]))
print(output, state)
