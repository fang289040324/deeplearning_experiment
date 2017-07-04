#!/usr/bin/env python
# coding=utf-8
import numpy as np
from env import Env

NUM_STATES = 8
GAMMA = 0.9


def play(model, weights):
    car_distance = 0
    env = Env(weights)

    state = env.reset()

    featureExpectations = np.zeros(len(weights))

    while True:
        car_distance += 1

        action = model.choose_action(state)

        state, reward, done = env.step(action)
        if car_distance > 100:
            featureExpectations += (GAMMA ** (car_distance - 101)) * np.array(state)
        if car_distance % 20000 == 0:
            break

    return featureExpectations


if __name__ == "__main__":
    pass
