

import numpy as np

# Example data array (5 samples, 4 features)
X = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20]
])

# Example p-values array
p_values = np.array([0.01, 0.2, 0.03, 0.5])

# Select columns where p-values are less than 0.05
selected_columns = X[:, p_values < 0.05]

print("Original array X:\n", X)
print("p-values:\n", p_values)
print("filtered p_values: {}".format(p_values < 0.05))
print("Selected columns where p-values < 0.05:\n", selected_columns)





# Example data array (5 samples, 4 features)
X = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20]
])

# Example p-values array
p_values = np.array([0.01, 0.2, 0.03, 0.5])

# Create the boolean mask
mask = p_values < 0.05

# Get the indices of the columns that meet the condition
selected_indices = np.where(mask)[0]

# Select columns where p-values are less than 0.05
X_new = X[:, mask]

# Print the original indices of the selected columns
print("Selected indices:", selected_indices)
print("X_new:\n", X_new)


