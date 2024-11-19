

import numpy as np

# Example data arrays
X = np.array([
    [1, -1, 0],
    [0, 1, 0],
    [2, 2, -5],
    [-1, -1, -1]
])

V = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
    [100, 110, 120]
])

# Select rows in V where the row sums in X are greater than 0
selected_rows = V[X.sum(axis=1) > 0, :]

print("Array X:\n", X)
print("Row sums of X:\n", X.sum(axis=1))
print("Boolean mask (X.sum(axis=1) > 0):\n", X.sum(axis=1) > 0)
print("Selected rows from V where row sums of X are > 0:\n", selected_rows)


