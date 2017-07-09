#!/usr/bin/env python
# coding=utf-8

import numpy as np

from playing import play
import ddpg
from cvxopt import matrix
from cvxopt import solvers

NUM_STATES = 8
FRAMES = 100000
GAMMA = 0.9


class IrlAgent:
    def __init__(self, randomFE, epsilon):
        self.randomPolicy = randomFE
        self.expertPolicy = self.get_expertPolicy()
        self.epsilon = epsilon
        self.randomT = np.linalg.norm(np.asarray(self.expertPolicy) - np.asarray(self.randomPolicy))
        self.policiesFE = {self.randomT: self.randomPolicy}
        print("Expert - Random at the Start (t) :: ", self.randomT)
        self.currentT = self.randomT
        self.minimumT = self.randomT
        # ddpg.init()

    def get_expertPolicy(self):
        i = 0
        x_main = np.load('../../data/bmw/x_main_all.npy')
        x_aux = np.load('../../data/bmw/x_aux_all.npy')
        x_main_reshape = x_main.reshape([-1, 24])
        x_aux_reshape = x_aux.reshape([-1, 15])
        result = np.concatenate([x_main_reshape, x_aux_reshape], axis=1).astype(np.float64)
        featureExpectations = np.zeros(result[0].shape)
        for state in result[:20000]:
            featureExpectations += (GAMMA ** i) * np.array(state)
        return featureExpectations

    def getRLAgentFE(self, W, i):
        # IRL_helper(W, path, self.num_frames, i)
        # saved_model = path
        # model = neural_net(self.num_states, [164, 150], saved_model)
        model = ddpg.train(W)
        print('======================================= play ========================================')
        return play(model, W)

    def policyListUpdater(self, W, i):
        tempFE = self.getRLAgentFE(W, i)
        hyperDistance = np.abs(np.dot(W, np.asarray(self.expertPolicy) - np.asarray(tempFE)))
        self.policiesFE[hyperDistance] = tempFE
        return hyperDistance

    def optimalWeightFinder(self):
        i = 1
        while True:
            W = self.optimization()
            # print("the distances  ::", self.policiesFE.keys())
            self.currentT = self.policyListUpdater(W, i)
            print("Current distance (t) is:: ", self.currentT)
            if self.currentT <= self.epsilon:
                break
            i += 1
        return W

    def optimization(self):
        m = len(self.expertPolicy)
        P = matrix(2.0 * np.eye(m), tc='d')
        print('=============================P rank:', np.linalg.matrix_rank(np.eye(m)), '=============================')
        q = matrix(np.zeros(m), tc='d')
        policyList = [self.expertPolicy]
        h_list = [1]
        for i in self.policiesFE.keys():
            policyList.append(self.policiesFE[i])
            h_list.append(1)
        policyMat = np.matrix(policyList)
        policyMat[0] *= -1
        print(policyMat)
        print('=============================G rank:', np.linalg.matrix_rank(policyMat), '=============================')
        G = matrix(policyMat, tc='d')
        h = matrix(-np.array(h_list), tc='d')
        sol = solvers.qp(P, q, G, h)

        weights = np.squeeze(np.asarray(sol['x']))
        norm = np.linalg.norm(weights)
        weights /= norm
        print('=============================== optimization weight', weights, '=============================')
        return weights


if __name__ == '__main__':
    randomPolicyFE = [7.69879930e+04, 1.50032536e+01, -1.61100000e+02, -5.76653179e+04,
                      5.34256075e+02, 7.90825000e+04, 3.23183892e+04, 1.29641108e+04,
                      5.34256075e+02, 5.34256075e+02, 2.00000000e+04, 0.00000000e+00,
                      -4.47398210e+02, -1.03479881e+03, -3.88418812e+02, -5.07880268e+05,
                      -1.02602039e+06, -9.39892266e+04, 1.40474547e+06, 7.69867794e+05,
                      4.47543820e+05, 1.42081138e+06, 8.13194481e+05, 3.05588346e+05,
                      2.00000000e+04, 0.00000000e+00, 0.00000000e+00, 2.00000000e+04,
                      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.38000000e+04,
                      3.87000000e+04, 1.00400000e+05, 8.71906988e+09, 8.86005299e+10,
                      1.00000000e+03, 7.69879930e+04, 0.00000000e+00]

    epsilon = 0.001
    irlearner = IrlAgent(randomPolicyFE, epsilon)
    print(irlearner.optimalWeightFinder())
