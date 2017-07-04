#!/usr/bin/env python
# coding=utf-8
import numpy as np
import time


class Env(object):
    def __init__(self, weights):
        self.x_main = np.load('../../data/bmw/x_main_all.npy')
        self.x_aux = np.load('../../data/bmw/x_aux_all.npy')
        x_main_reshape = self.x_main.reshape([-1, 24])
        x_aux_reshape = self.x_aux.reshape([-1, 15])
        self.result = np.concatenate([x_main_reshape, x_aux_reshape], axis=1).astype(np.float64)
        self.weights = weights
        self.action_dim = 2
        self.sub_result = None
        self.index = 0

    def reset(self):
        return self.result[0].astype(np.float64)

    """
    a (VelocityAbs, YawAbs)
    """
    def step(self, a):
        result = self.result
        print(result.shape)
        velocitys = result[:, 0].reshape([-1, 1])
        yaws = result[:, 3].reshape([-1, 1])

        A = np.concatenate([yaws, velocitys], axis=1)
        A = A.astype(np.float64)
        print('A:', A[0])
        a_index = np.argwhere((A == a).all(1))
        if a_index.shape[0]:
            state = result[a_index + 1]
        else:
            state = np.ones([self.observation_space(), ]) * -1
        reward = np.dot(self.weights, state)
        done = None

        return state, reward, done

    def get_memory(self, m):
        # if self.index + self.n <= self.result.shape[0]:
        #     self.sub_result = self.result[self.index: self.index + self.n]
        #     self.index += 1
        # elif self.index < self.result.shape[0]:
        #     self.sub_result = self.result[self.index:]
        # else:
        #     self.index = 0
        #     self.sub_result = self.result[self.index: self.index + self.n]
        #     self.index += self.n
        #
        # for i in range(self.sub_result.shape[0] - 1):
        #     d = self.sub_result[i]
        #     s_ = self.sub_result[i + 1]
        #     m.store_transition(d, np.array([d[0], d[3]]), np.dot(self.weights, d), s_)

        d = self.result[self.index]
        self.index += 1
        if self.index >= self.result.shape[0]:
            self.index = 0
        m.store_transition(d, np.array([d[0], d[3]]), np.dot(self.weights, d), self.result[self.index + 1])

    def observation_space(self):
        return self.result.shape[-1]

    def action_space(self):
        return self.action_dim

    def render(self):
        pass

    def action_bound(self):
        # TODO
        return 1


if __name__ == '__main__':
    x_main = np.load('../../data/bmw/x_main_all.npy')
    print(x_main[0, 0].astype(np.float64))
    print(x_main[0, 0])
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
