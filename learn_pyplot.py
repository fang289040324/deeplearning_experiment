#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, num=50)
plt.plot(x, np.sin(x), 'r-.', x, np.cos(x), 'g--')
plt.show()

x = np.linspace(0, 10)
plt.subplot(211)
plt.plot(x, np.sin(x), 'r--')
plt.subplot(212)
plt.plot(x, np.cos(x), 'g-.')
plt.show()

x = np.linspace(0, 10)
plt.scatter(x, np.sin(x))
plt.show()

x = np.random.normal(0, 100, 10000)
y = np.random.normal(0, 100, 10000)
size = np.random.rand(1000) * 10  # 每个点的大小
colors = np.random.rand(10000)
plt.scatter(x, y, size, colors)
plt.colorbar()
plt.show()

x = np.random.normal(0, 10, 10000)
plt.hist(x, 20)  # 第二个参数是条的个数
plt.show()

x = np.linspace(0, 5)
plt.plot(x, np.sin(x), 'r--', label='sin(x)')
plt.plot(x, np.cos(x), 'g-.', label='cos(x)')
plt.legend()
plt.show()
