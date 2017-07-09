#!/usr/bin/env python
# coding=utf-8
import numpy as np
from env import Env

GAMMA = 0.9


def play(actor, weights):
    car_distance = 0
    env = Env(weights)

    state = env.reset()

    featureExpectations = np.zeros(len(weights))

    while True:
        car_distance += 1

        action = actor.target_model.predict(state.reshape(1, state.shape[0]))
        state, reward, done = env.step(action)
        if car_distance % 1000 == 0:
            print('======================action:', action, 'reward:', reward, '========================')
        if car_distance > 100:
            featureExpectations += (GAMMA ** (car_distance - 101)) * np.array(state)
        if car_distance % 2000 == 0:
            break

    return featureExpectations + np.random.randn(state.shape[0])


if __name__ == "__main__":
    pass
