from sklearn import preprocessing as pre
import numpy as np

a = np.asarray([1, 2, 3, 4, 66, 44, 111, 55], dtype=np.float32).reshape([-1, 4]).T
print(a)
pre_a = pre.StandardScaler().fit(a)

print(pre_a.transform(a))
print(pre.scale(a))
print(pre_a.transform(np.asarray([88, 7, 5, 44, 1, 2, 8, 5], np.float32).reshape([-1, 4]).T))
