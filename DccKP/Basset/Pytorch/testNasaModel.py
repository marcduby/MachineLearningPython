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
file_model = dir_data + "Data/Broad/Basset/Nasa/ampt2d_cnn_900_best.th"

# load the model
model = torch.load(file_model)

# print the model
print("the model is \n{}".format(model))


