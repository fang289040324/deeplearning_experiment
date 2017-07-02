#!/usr/bin/env python
# coding=utf-8

import gym
import time

# env = gym.make('CartPole-v1')
# env = gym.make('MountainCar-v0')
env = gym.make('Enduro-v4')
for i in range(20):
    observation = env.reset()
    for j in range(100):
        start_t = time.time()
        env.render()
        print(observation)
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        print(reward)
        print('cost time: ', (time.time() - start_t))
        if done:
            print("Episode finished after {} timesteps".format(j + 1))
            break
    print('------------------------------------------------------')


env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


from gym import spaces

space = spaces.Discrete(8)  # Setwith8elements{0,1,2,...,7}
x = space.sample()
assert space.contains(x)
assert space.n == 8

from gym import envs
print(envs.registration.registry.all())
