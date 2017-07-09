import keras as k
import keras.backend as K
import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        S = k.layers.Input(shape=[state_size])
        A = k.layers.Input(shape=[action_dim], name='action2')
        w1 = k.layers.Dense(1000, activation='tanh')(S)
        h1 = k.layers.Dense(600, activation='tanh')(w1)
        h1 = k.layers.Dense(300, activation='tanh')(h1)
        a1 = k.layers.Dense(300, activation='tanh')(A)
        a1 = k.layers.Dense(300, activation='tanh')(a1)
        h2 = k.layers.add([h1, a1])
        h3 = k.layers.Dense(100, activation='tanh')(h2)
        V = k.layers.Dense(action_dim, activation='linear')(h3)
        model = k.models.Model(inputs=[S, A], outputs=V)
        adam = k.optimizers.Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S
