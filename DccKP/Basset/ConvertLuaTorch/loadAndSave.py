# imports
import torch
from torch import nn
import twobitreader
from twobitreader import TwoBitFile
from torch.utils.serialization import load_lua

print("got pytorch version of {}".format(torch.__version__))

# set the code and data directories
dir_code = "/Users/mduby/Code/WorkspacePython/"
dir_data = "/Users/mduby/Data/Broad/"
# dir_code = "/home/javaprog/Code/PythonWorkspace/"
# dir_data = "/home/javaprog/Data/Broad/"

# import relative libraries
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Basset/')
import dcc_basset_lib

# file input
file_input = dir_data + "Magma/Common/part-00011-6a21a67f-59b3-4792-b9b2-7f99deea6b5a-c000.csv"
file_model_weights = dir_data + 'Basset/Nasa/ampt2d_cnn_900_best_cpu.th'
file_twobit = dir_data + 'Basset/TwoBitReader/hg19.2bit'

# LOAD THE MODEL
# load the weights
state_dict = load_lua(file_model_weights)

print(state_dict.__class__)
print(state_dict)
print(state_dict.model)
print(type(state_dict.model))
# print(help(state_dict.model))
print(state_dict.model.__class__)

model = state_dict.model
# model.evaluate()

# SUCCESS - print the model weights
for i in (0, 1):
    print("matrix structure {} has length {} and shape {}".format(i, len(state_dict.model.parameters()[i]), type(state_dict.model.parameters()[i])))
# print(len(state_dict.model.parameters()[0]))
# print(type(state_dict.model.parameters()[0]))
# print(state_dict.model.parameters()[0])

# array item 0 is weights, array item 1 is bias
for item in state_dict.model.parameters()[1]:
    print("item type {} and shape: {}".format(type(item), item.shape))

# major_index = 1
for major_index in (0, 1):
    minor_index = 25
    tensor = state_dict.model.parameters()[major_index][minor_index]
    print("tensor {}-{} has shape {} and values {}".format(major_index, minor_index, tensor.shape, tensor))

count = 0
print("model.modules is of type {}".format(type(model.modules)))
for module in model.modules:
    print("({}) model name {} and type {} and weights {}".format(count, module, type(module), module.bias.shape))
    count = count + 1
# print(model.layer[0])
# print(help(state_dict.model))
# print(state_dict.model.state_dict())
# model.evaluate()
# new_dict = model.state_dict()

# model.forward(torch.Tensor(1, 2, 3))

# torch.save(state_dict.model, dir_data + 'Basset/Marc/LoadLua/dude_model.pth')


# state_dict.model.forward(torch.FloatTensor(1, 2, 3))


# model = dcc_basset_lib.load_basset_model_from_state_dict(state_dict.model)

# model = state_dict.Model

# model.evaluate()
# new_dict = model.state_dict()


# # save using pickle format
# torch.save(new_dict, '/home/javaprog/dude.pth')


