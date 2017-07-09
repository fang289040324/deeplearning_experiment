#!/usr/bin/env python
# coding=utf-8
import numpy as np
import time


class Env(object):
    def __init__(self):
        self.x_main = np.load('../../data/bmw/x_main_all.npy')
        self.x_aux = np.load('../../data/bmw/x_aux_all.npy')
        x_main_reshape = self.x_main.reshape([-1, 24])
        x_aux_reshape = self.x_aux.reshape([-1, 15])
        self.result = np.concatenate([x_main_reshape, x_aux_reshape], axis=1).astype(np.float64)
        self.action_dim = 2
        self.sub_result = None
        self.index = 0

        velocitys = self.result[:, 0].reshape([-1, 1])
        yaws = self.result[:, 3].reshape([-1, 1])

        self.A = np.concatenate([velocitys, yaws], axis=1)
        self.A = self.A.astype(np.float64)

        self.A_index = {}
        for i, d in enumerate(self.A):
            self.A_index[i] = d

        self.A_index = sorted(self.A_index.items(), key=lambda x: x[1][1])

        self.A_0_min = np.min(self.A[:, 0])
        self.A_0_max = np.max(self.A[:, 0])
        self.A_1_min = np.min(self.A[:, 1])
        self.A_1_max = np.max(self.A[:, 1])

        self.find_list_temp = []

    def reset(self):
        return self.result[0].astype(np.float64), 0

    """
    a (VelocityAbs, YawAbs)
    """
    def step(self, a, index):
        if index == self.result.shape[0] - 1:
            index = 0
        state = self.result[index].reshape([-1])
        state_ = self.result[index + 1].reshape([-1])
        reward = -np.linalg.norm(np.array([state[0], state[3]]) - a) + 1

        done = None

        return state_, reward, done, index + 1

    def observation_space(self):
        return self.result.shape[-1]

    def action_space(self):
        return self.action_dim

    def render(self):
        pass


if __name__ == '__main__':
    pass