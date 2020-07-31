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


# print the row numbers
print("the row size of the weight matrix is {}".format(weight_matrix.shape[0]))
print("the column size of the weight matrix is {}".format(weight_matrix.shape[1]))


# print the information on tensors
print("the weight tensor is of dtype {}".format(weight_matrix.dtype))
print("the weight tensor is of device {}".format(weight_matrix.device))
print("the weight tensor is of layout {}".format(weight_matrix.layout))

# get random tensor
rand_tensor = torch.rand(3, 5)
print("the random tensor is {}".format(rand_tensor))

# get the identity matrix
eye_tensor = torch.eye(4)
print("the identity tensor is {}".format(eye_tensor))
