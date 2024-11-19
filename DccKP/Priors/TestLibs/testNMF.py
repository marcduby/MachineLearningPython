

import numpy as np
from sklearn.decomposition import NMF

# Example data matrix
V = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])

# Define the number of components
n_components = 2

# Initialize and fit the NMF model
model = NMF(n_components=n_components, init='random', random_state=0)
W = model.fit_transform(V)
H = model.components_

# Resulting matrices
print("Matrix V:")
print(V)
print("\nMatrix W:")
print(W)
print("\nMatrix H:")
print(H)
print("\nReconstructed V (W @ H):")
print(np.dot(W, H))



