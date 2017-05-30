#!/usr/bin/env python
# coding=utf-8

import csv

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

with open('C:\\Users\\fangsw\\PycharmProjects\\experiment\\BMWdata1206n.csv') as f:
    reader = csv.DictReader(f)
    map1 = {}
    for row in reader:
        if row['AgentID'] in map1.keys():
            map1[row['AgentID']].append((row['PositionAbsX'], row['PositionAbsY'], row['Timestamp']))
        else:
            map1[row['AgentID']] = [(row['PositionAbsX'], row['PositionAbsY'], row['Timestamp'])]
    print(len(map1))
    print(len(map1.keys()))

    num_map = {}
    for data in map1.keys():
        num_map[data] = len(map1[data])

    num_list = sorted(list(num_map.items()), key=lambda item: item[1], reverse=True)

    print(num_list[0])
    print(num_list[1])
    print(num_list[2])

    xl = []
    yl = []
    zl = []
    l = []
    # for data in map1['0']:
    #     xl.append(data[0])
    #     yl.append(data[1])
    #     zl.append(data[2])
    #     l.append('r')
    for data in map1['1730']:
        xl.append(data[0])
        yl.append(data[1])
        zl.append(data[2])
        l.append('g')
    # for data in map1['89']:
    #     xl.append(data[0])
    #     yl.append(data[1])
    #     zl.append(data[2])
    #     l.append('y')
        # for data in map1['4']:
        #     xl.append(data[0])
        #     yl.append(data[1])
        #     l.append('b')
        # for data in map1['15']:
        #     xl.append(data[0])
        #     yl.append(data[1])
        #     l.append('r')
        # for data in map1['165']:
        #     xl.append(data[0])
        #     yl.append(data[1])
        #     l.append('r')

# 产生测试数据
x = np.array(xl, dtype=np.float32)
y = np.array(yl, dtype=np.float32)
z = np.array(zl, dtype=np.float32)

ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程

# 将数据点分成三部分画，在颜色上有区分度
ax.scatter(x, y, z, c=l)  # 绘制数据点

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()
