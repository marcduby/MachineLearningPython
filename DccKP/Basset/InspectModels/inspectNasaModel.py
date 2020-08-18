# imports
import torch
from torch import nn
import twobitreader
from twobitreader import TwoBitFile

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
labels_file = dir_data + '/Basset/Production/nasa_labels.txt'

# LOAD THE MODEL
# load the weights
nasa_model = dcc_basset_lib.load_nasa_model(file_model_weights)

# make the model eval
nasa_model.eval()

# better summary
print(nasa_model)

# print the weights
# print("\n\nprinting weight shapes")
# for index in range(0, len(basset_model) - 1):
#     model_module = basset_model[index]
#     if hasattr(model_module, 'weight'):
#         print("({}) params for layer name {} has type {} and weights {}".format(index, model_module, type(model_module), model_module.weight.shape))


# print the weights for layer 5 (conv2d)
