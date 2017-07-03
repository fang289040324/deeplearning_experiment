import tensorflow as tf
import numpy as np
from env import Env

np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 70
MAX_EP_STEPS = 400
LR_A = 0.01
LR_C = 0.01
GAMMA = 0.9
TAU = 0.01
REPLACE_ITER_A = 500
REPLACE_ITER_C = 300
MEMORY_CAPACITY = 7000
BATCH_SIZE = 32

S = None
R = None
S_ = None

RENDER = False


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            self.a = self._build_net(S, scope='eval_net', trainable=True)
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='l1', trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w, bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')
        return scaled_a

    def learn(self, s):
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.a, feed_dict={S: s})[0]

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # self.a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net',
                                      trainable=False)

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b,
                                    trainable=trainable)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})

        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


def train(w):
    global RENDER, S, R, S_
    env = Env(w)

    state_dim = env.observation_space()
    action_dim = env.action_space()
    action_bound = env.action_bound()

    with tf.name_scope('S'):
        S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
    with tf.name_scope('R'):
        R = tf.placeholder(tf.float32, [None, 1], name='r')
    with tf.name_scope('S_'):
        S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

    sess = tf.Session()

    actor = Actor(sess, action_dim, action_bound, LR_A, REPLACE_ITER_A)
    critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
    actor.add_grad_to_graph(critic.a_grads)

    sess.run(tf.global_variables_initializer())

    M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

    var = 3

    for i in range(MAX_EPISODES):
        s = env.reset()
        print(s)
        ep_reward = 0

        for j in range(MAX_EP_STEPS):

            if RENDER:
                env.render()

            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)
            print('a: ', a)
            s_, r, done = env.step(a)

            M.store_transition(s, a, r / 10, s_)
            print(M.pointer)
            if M.pointer > MEMORY_CAPACITY:
                var *= .9995
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :state_dim]
                b_a = b_M[:, state_dim: state_dim + action_dim]
                b_r = b_M[:, -state_dim - 1: -state_dim]
                b_s_ = b_M[:, -state_dim:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_reward += r

            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var,)
                if ep_reward > -1000:
                    RENDER = True
                break

if __name__ == '__main__':
    train(np.ones([39, ]))
