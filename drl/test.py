import numpy as np


class OU(object):
    @staticmethod
    def function(x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)


x = -1
x_t = 0

for _ in range(1000):
    noise = OU.function(x, 0, 1, 0.3)
    print('noise: ', noise)
    x_t = x + noise
    print('x_t: ', x_t)
