#!/usr/bin/env python
# coding=utf-8

from cvxopt import solvers, matrix

P = matrix([[1., 0.], [0., 0.]])
q = matrix([3., 4.])
G = matrix([[-1., 0., -1., 2., 3.], [-1., -1., -1., -1., -1.]])
h = matrix([0., 0., -15., 100., 80.])

sol = solvers.qp(P, q, G, h)
print(sol['x'])
