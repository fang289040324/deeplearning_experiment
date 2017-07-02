import numpy as np
import time


class Env(object):
    def __init__(self, weights):
        self.x_main = np.load('./data/bmw/x_main_all.npy')
        self.x_aux = np.load('./data/bmw/x_aux_all.npy')
        x_main_reshape = x_main.reshape([-1, 24])
        x_aux_reshape = x_aux.reshape([-1, 15])
        self.result = np.concatenate([x_main_reshape, x_aux_reshape], axis=1)
        self.weights = weights
        self.action_dim = 2

    def reset(self):
        return self.result[0]

    """
    a (YawAbs, VelocityAbs)
    """
    def step(self, a):
        velocitys = self.result[:, 0].reshape([-1, 1])
        yaws = self.result[:, 3].reshape([-1, 1])

        A = np.concatenate([yaws, velocitys], axis=1)
        a_index = np.argwhere((A == a).all(1))

        state = self.result[a_index + 1]
        reward = np.dot(self.weights, state)
        done = None

        return state, reward, done

    def observation_space(self):
        return len(self.result[0])

    def action_space(self):
        return self.action_dim


if __name__ == '__main__':
    x_main = np.load('../../data/bmw/x_main_all.npy')
    x_aux = np.load('../../data/bmw/x_aux_all.npy')
    x_main_reshape = x_main.reshape([-1, 24])
    x_aux_reshape = x_aux.reshape([-1, 15])
    print(x_main_reshape.shape, x_aux_reshape.shape)
    result = np.concatenate([x_main_reshape, x_aux_reshape], axis=1)
    print(result.shape)
    print(result[0].shape)
    # print(result[0])
    start = time.time()
    print(np.argwhere((result == result[4598]).all(1)))
    print(time.time() - start)
