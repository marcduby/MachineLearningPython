# imports
import torch
from torch import nn
import twobitreader
from twobitreader import TwoBitFile
from torch.utils.serialization import load_lua

print("got pytorch version of {}".format(torch.__version__))

# set the code and data directories
# dir_code = "/Users/mduby/Code/WorkspacePython/"
# dir_data = "/Users/mduby/Data/Broad/"
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/home/javaprog/Data/Broad/"

# import relative libraries
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Basset/')
import dcc_basset_lib

# file input
file_input = dir_data + "Magma/Common/part-00011-6a21a67f-59b3-4792-b9b2-7f99deea6b5a-c000.csv"
file_model_weights = dir_data + 'Basset/Nasa/ampt2d_cnn_900_best_cpu.th'
new_nasa_model_weights = dir_data + 'Basset/Marc/Test/ampt2d_cnn_900_best_p041.pth'
file_twobit = dir_data + 'Basset/TwoBitReader/hg19.2bit'

# LOAD THE MODEL
# load the weights
state_dict = load_lua(file_model_weights)

# get the models
lua_model = state_dict.model
nasa_model = dcc_basset_lib.load_nasa_model_from_state_dict(None)

# count = 0
# print("model.modules is of type {}".format(type(model.modules)))
# for module in model.modules:
#     print("({}) model name {} and type {} and weights {}".format(count, module, type(module), module.bias.shape))
#     count = count + 1

# for i in (0, 1):
#     module = model.modules[i]
#     print("({}) model name {} and type {} and weights {}".format(i, module, type(module), module.weight.shape))

# print layer weight before setting
# index = 1
# module = nasa_model[index]
# print("layer {} has shape {} and weights {}".format(index, module.weight.shape, module.weight))
# module.weight = nn.Parameter(model.modules[index].weight)
# module.bias = nn.Parameter(model.modules[index].bias)
# print("layer {} has shape {} and weights {}".format(index, module.weight.shape, module.weight))

# need to transpose dimensions 2 and 3 for the con2d layers
for index in (0, 4, 8, 12):
    old_module = lua_model.modules[index]
    # print("({}) model name {} and type {} and weights {}".format(index, old_module, type(old_module), old_module.weight.shape))
    nasa_module = nasa_model[index]
    print("({}) setting params for layer name {} has type {} and weights {}".format(index, nasa_module, type(nasa_module), nasa_module.weight.shape))
    with torch.no_grad():
        nasa_module.weight = nn.Parameter(torch.transpose(lua_model.modules[index].weight, 2, 3))
        nasa_module.bias = nn.Parameter(lua_model.modules[index].bias)

# layer that do not need transposition
for index in (1, 5, 9, 13, 18, 22):
    old_module = lua_model.modules[index]
    # print("({}) model name {} and type {} and weights {}".format(index, old_module, type(old_module), old_module.weight.shape))
    nasa_module = nasa_model[index]
    print("({}) setting params for layer name {} has type {} and weights {}".format(index, nasa_module, type(nasa_module), nasa_module.weight.shape))
    with torch.no_grad():
        nasa_module.weight = nn.Parameter(lua_model.modules[index].weight)
        nasa_module.bias = nn.Parameter(lua_model.modules[index].bias)

# linear sub layers
for index in (17, 21, 25):
    old_module = lua_model.modules[index]
    # print("({}) model name {} and type {} and weights {}".format(index, old_module, type(old_module), old_module.weight.shape))
    nasa_module = nasa_model[index][1]
    print("({}) setting params for layer name {} has type {} and weights {}".format(index, nasa_module, type(nasa_module), nasa_module.weight.shape))
    with torch.no_grad():
        nasa_module.weight = nn.Parameter(lua_model.modules[index].weight)
        nasa_module.bias = nn.Parameter(lua_model.modules[index].bias)

# save the network
nasa_model.eval()
print("saving to file {}".format(new_nasa_model_weights))
torch.save(nasa_model.state_dict(), new_nasa_model_weights)
