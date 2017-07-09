import keras.backend as K
import keras as k
import tensorflow as tf


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        S = k.layers.Input(shape=[state_size])
        h0 = k.layers.Dense(600, activation='relu')(S)
        h1 = k.layers.Dense(600, activation='relu')(h0)
        h1 = k.layers.Dense(600, activation='relu')(h1)
        h1 = k.layers.Dense(600, activation='relu')(h1)
        h1 = k.layers.Dense(100, activation='tanh')(h1)
        yaw = k.layers.Dense(1, activation='tanh', kernel_initializer=k.initializers.glorot_normal())(h1)
        yaw_ = k.layers.Lambda(lambda yaw: yaw * 100)(yaw)
        velocitys = k.layers.Dense(1, activation='sigmoid', kernel_initializer=k.initializers.glorot_normal())(h1)
        velocitys_ = k.layers.Lambda(lambda v: v * 60)(velocitys)
        # V = k.layers.concatenate([velocitys, yaw])
        V = k.layers.concatenate([velocitys_, yaw_])
        model = k.models.Model(inputs=S, outputs=V)
        return model, model.trainable_weights, S
