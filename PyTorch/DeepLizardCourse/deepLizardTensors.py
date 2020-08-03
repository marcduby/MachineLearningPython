# imports
import torch

print("using torch version {}".format(torch.__version__))


# create a tensor
tensor01 = torch.rand(3, 5, 2)

# get the type and shape of a tensor
print("the tensor has type {} and shape {} and rank {}".format(type(tensor01), tensor01.shape, len(tensor01.shape)))

# reshape the tensor (changes shape but not underlying elements)
tensor02 = tensor01.reshape(3, 10)
print("the tensor has type {} and shape {} and rank {}".format(type(tensor02), tensor02.shape, len(tensor02.shape)))

# for images, tensors usually [batch_size, color_channel, height, width]


