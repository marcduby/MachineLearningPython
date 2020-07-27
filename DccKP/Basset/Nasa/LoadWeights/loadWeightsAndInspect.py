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

print(lua_model)

for i in (0, 1, 4, 5, 8, 9, 12, 13, 17, 18, 21, 22, 25):
    module = lua_model.modules[i]
    print("({}) model name {} and type {} and weights {}".format(i, module, type(module), module.weight.shape))

print()

for i in (0, 1, 4, 5, 8, 9, 12, 13, 17, 18, 21, 22, 25):
    module = lua_model.modules[i]
    print("({}) model name {} and type {} and bias {}".format(i, module, type(module), module.bias.shape))

# for i in (17, 21, 25):
#     module = lua_model.modules[i]
#     print("({}) model name {} and type {} and weights {}".format(i, module, type(module), module.weight.shape))
#     print("({}) model name {} and type {} and bias {}".format(i, module, type(module), module.bias.shape))

for i in (1, 5, 9, 13, 18, 22):
    module = lua_model.modules[i]
    print("({}) model name {} and type {} and mean {}".format(i, module, type(module), module.running_mean.shape))


