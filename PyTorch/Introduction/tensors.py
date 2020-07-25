# imports
import torch

print("the torch version is {}".format(torch.__version__))

# create 2 tensors
in_features = torch.tensor([1, 2, 3, 4, 5])
weight_matrix = torch.tensor([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7]
])

# print the shapes
print("the input features have shape {}".format(in_features.shape))
print("the weight matrix features have shape {}".format(weight_matrix.shape))

# multiply the tensors and print the result
result = weight_matrix.matmul(in_features)
print("the result is {}".format(result))
print("the result has shape {}".format(result.shape))

