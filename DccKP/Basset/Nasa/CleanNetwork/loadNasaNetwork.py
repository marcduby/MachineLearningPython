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
# file_model_weights = dir_data + 'Basset/Nasa/ampt2d_cnn_900_best_cpu.th'
file_model_weights = dir_data + 'Basset/Marc/LoadLua/dude_model.pth'
file_twobit = dir_data + 'Basset/TwoBitReader/hg19.2bit'

# LOAD THE MODEL
# load the weights
# state_dict = load_lua(file_model_weights)
# pretrained_model_reloaded_th = dcc_basset_lib.load_nasa_model_from_state_dict(state_dict)
pretrained_model_reloaded_th = dcc_basset_lib.load_nasa_model_from_state_dict(None)
# pretrained_model_reloaded_th = dcc_basset_lib.load_nasa_model_from_state_dict(file_model_weights)

# make the model eval
# pretrained_model_reloaded_th.eval()

# better summary
print(pretrained_model_reloaded_th)

 
# access the 3rd layer
index = 2
for index, layer in enumerate(pretrained_model_reloaded_th.modules()):
    # layer = pretrained_model_reloaded_th.modules()[2]
    print("module of index {} is of type {} and str {}".format(index, type(layer), layer))




