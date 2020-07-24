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

# load the model
# model_file = dir_data + "Basset/Nasa/ampt2d_cnn_900_best.th"
model_file = dir_data + "Basset/Model/ampt2d_cnn_900_best_cpu.th"
state_dict = load_lua(model_file)

# print("the dict model is \n{}".format(state_dict))

# model = dcc_basset_lib.load_basset_model_from_state_dict(state_dict)

print("the state_dict type is \n{}".format(type(state_dict)))
print("the model type is \n{}".format(type(state_dict.model)))
dict_keys = list(state_dict.keys())

print("load lua aboject keys \n{}".format(dict_keys))
# print("the model is \n{}".format(state_dict.model))

print("object \n{}".format(state_dict.model))


