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
        return self.result[0].astype(np.float64)

    """
    a (VelocityAbs, YawAbs)
    """
    def step(self, a):
        start = time.time()
        result = self.result
        state = np.ones([self.observation_space(), ]) * -1
        print(a, self.A_0_min, self.A_0_max, self.A_1_min, self.A_1_max)
        if a[0] > self.A_0_max or a[0] < self.A_0_min or a[1] > self.A_1_max or a[1] < self.A_1_min:
            reward = -1
        else:
            min_d = 999
            a_index = 0
            is_find = False
            for i, aa in enumerate(self.A):
                d = np.linalg.norm(aa - a)
                if min_d > d and d < 0.1 and i < self.A.shape[0]:
                    min_d = d
                    a_index = i
                    is_find = True
                    break

            if is_find:
                state = result[a_index + 1].reshape([-1])
                reward = state[0] * np.cos(state[4]) - state[1] * np.sin(state[4]) + min(10, state[23])
            else:
                reward = -1

        print(a_index)
        print(time.time() - start)
        done = None

        return state, reward, done

    # def get_memory(self, m):
    #     d = self.result[self.index]
    #     # print(np.dot(self.weights, d))
    #     self.index += 1
    #     if self.index + 1 >= self.result.shape[0]:
    #         self.index = 0
    #
    #     m.add(d, np.array([d[0], d[3]]), np.dot(self.weights, d), self.result[self.index + 1], None)

    def find(self, a, start, end):
        if end - start == 1:
            return
        scope_0 = 0.001
        scope_1 = 0.00001
        temp_list = self.A_index
        mid = int((end - start) / 2) + start
        mid_ = temp_list[mid][1][1]
        a_ = a[1]
        print(start, end)
        print(mid_, np.abs(a_ - mid_) < scope_1)

        if a_ > mid_ and np.abs(a_ - mid_) > scope_1:
            self.find(a, mid, end)
        elif a_ < mid_ and np.abs(a_ - mid_) > scope_1:
            self.find(a, start, mid)
        elif a_ > mid_ and np.abs(a_ - mid_) < scope_1:
            self.find_list_temp.append(temp_list[mid])
            self.find(a, mid, end)
        elif a_ < mid_ and np.abs(a_ - mid_) < scope_1:
            self.find_list_temp.append(temp_list[mid])
            self.find(a, start, mid)
        else:
            self.find_list_temp.append(temp_list[mid])

        print(self.find_list_temp)

        for d in self.find_list_temp:
            if np.abs(a[0] - d[1][0]) < scope_0:
                return d[0]

    def observation_space(self):
        return self.result.shape[-1]

    def action_space(self):
        return self.action_dim

    def render(self):
        pass


if __name__ == '__main__':
    # x_main = np.load('../../data/bmw/x_main_all.npy')
    # print(x_main[0, 0].astype(np.float64))
    # print(x_main[0, 0])
    # x_aux = np.load('../../data/bmw/x_aux_all.npy')
    # x_main_reshape = x_main.reshape([-1, 24])
    # x_aux_reshape = x_aux.reshape([-1, 15])
    # print(x_main_reshape.shape, x_aux_reshape.shape)
    # result = np.concatenate([x_main_reshape, x_aux_reshape], axis=1)
    # print(result.shape)
    # print(result[0].shape)
    # # print(result[0])
    # start = time.time()
    # print(np.argwhere((result == result[4598]).all(1)))
    # print(time.time() - start)

    env = Env()
    start = time.time()
    index = env.find([12.061, -2.983], 0, len(env.A_index))
    print(time.time() - start)
    print(index, env.result[index])
    for i, d in enumerate(env.A_index):
        if d[0] == 0:
            print(i, d)
    print('==================================================')
    print(env.step([12.061, -2.983]))
