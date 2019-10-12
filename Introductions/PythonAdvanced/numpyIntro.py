
# numpy test
import numpy as np
# import yaml

# print(help('modules'))
print(dir(np))
# print(dir(yaml))

# create arrays
one_d1 = np.array([1, 2, 3])
one_d2 = np.array([4, 5, 6])
two_d1 = np.array([[1, 2, 3, 4], [7, 6, 5, 4]])

print("array 1 is: {}".format(one_d1))
print("array 2 is: {}".format(one_d2))
print("array multiplication is: {}".format(one_d1 * one_d2))
print("array 2d is: {}".format(two_d1))

# get dimension
print()
print("dimension of 1d is {}".format(one_d1.ndim))
print("dimension of 2d is {}".format(two_d1.ndim))

# get shape
print()
print("dimension of 1d is {}".format(one_d1.shape))
print("dimension of 2d is {}".format(two_d1.shape))
