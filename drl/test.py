import tensorflow as tf
import keras as k

state = tf.Variable(0)
one = tf.constant(1)

new_val = tf.add(state, one)

update = tf.assign(state, new_val)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for _ in range(4):
        sess.run(update)
        print(state.eval())
