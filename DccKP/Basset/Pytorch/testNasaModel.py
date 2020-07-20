# imports
import twobitreader
from twobitreader import TwoBitFile
import numpy as np 
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder
import csv

# file locations
dir_data = "/Users/mduby/"
dir_code = "/Users/mduby/Code/WorkspacePython/"
file_model_weights = dir_data + "Data/Broad/Basset/Nasa/ampt2d_cnn_900_best.th"

# import relative libraries
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Basset/')
import dcc_basset_lib


# load the model
# model = torch.load(file_model_weights)

# load the model using the library
pretrained_model_reloaded_th = dcc_basset_lib.load_basset_model(file_model_weights)

# print the model
print("the model is \n{}".format(model))


