# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# copied from github below for use in my project at work
# https://github.com/kipoi/models/blob/master/Basset/pretrained_model_reloaded_th.py
# see paper at
# http://kipoi.org/models/Basset/

# imports
import torch
from torch import nn
import twobitreader
from twobitreader import TwoBitFile
# from torch.utils.serialization import load_lua

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
file_model_weights = dir_data + 'Basset/Production/nasa_ampt2d_cnn_900_best_p041.pth'
file_twobit = dir_data + 'Basset/TwoBitReader/hg19.2bit'

# LOAD THE MODEL
# load the weights
# state_dict = load_lua(file_model_weights)
# pretrained_model_reloaded_th = dcc_basset_lib.load_nasa_model_from_state_dict(state_dict.model)
pretrained_model_reloaded_th = dcc_basset_lib.load_nasa_model(file_model_weights)

# make the model eval
pretrained_model_reloaded_th.eval()

# better summary
print(pretrained_model_reloaded_th)

# print all weights and biases
print("\nprinting weights shape")
for index in range(0, 23):
    module = pretrained_model_reloaded_th[index]
    if hasattr(module, 'weight'):
        print("({}) layer {} has: ".format(index, module))
        print("\t== weights shape {} and bias shape {}".format(module.weight.shape, module.bias.shape))

# print a layer
print("\nprinting linear layers weights shape")
for index in (17, 21, 25):
    module = pretrained_model_reloaded_th[index][1]
    # print("({}) layer {} has weights shape {} and weights {}".format(index, module, module.weight.shape, module.weight))
    print("({}) layer {} has weights shape {} and bias shape {}".format(index, module, module.weight.shape, module.bias.shape))

