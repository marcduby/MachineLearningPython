# imports
import torch
from torch import nn

print("got pytorch version of {}".format(torch.__version__))

# set the code and data directories
# dir_code = "/Users/mduby/Code/WorkspacePython/"
# dir_data = "/Users/mduby/Data/Broad/"
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/home/javaprog/Data/Broad/"
file_test = "Basset/Marc/Test/untrained_nasa_model01.pth"

# import relative libraries
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Basset/')
import dcc_basset_lib

# LOAD THE MODEL
pretrained_model_reloaded_th = dcc_basset_lib.load_nasa_model_from_state_dict(None)

# print the network
print(pretrained_model_reloaded_th)

# print the network weights
for index in [0, 1, 4, 5, 8, 9, 12, 13, 18, 22]:
    print("the {} layer {} weight shape is {}.".format(index, pretrained_model_reloaded_th[index], pretrained_model_reloaded_th[index].weight.shape))
print()
for index in [0, 1, 4, 5, 8, 9, 12, 13, 18, 22]:
    print("the {} layer {} bias shape is {}.".format(index, pretrained_model_reloaded_th[index], pretrained_model_reloaded_th[index].bias.shape))
print("the layer shape is {}.".format(pretrained_model_reloaded_th[25][1].weight.shape))

# print the layer
index = 13
print("layer {} has shape {} and data \n{}".format(index, pretrained_model_reloaded_th[index].weight.shape, pretrained_model_reloaded_th[index].weight))

# set a layer weight to ones
# ones = torch.ones(300, 300, 5, 1)
# ones = torch.zeros(300)
# pretrained_model_reloaded_th[index].weight = nn.Parameter(ones)

# print the layer
print("layer {} has shape {} and data \n{}".format(index, pretrained_model_reloaded_th[index].weight.shape, pretrained_model_reloaded_th[index].weight))

# save the network
model_save_file = dir_data + file_test
pretrained_model_reloaded_th.eval()
print("saving to file {}".format(model_save_file))
torch.save(pretrained_model_reloaded_th.state_dict(), model_save_file)

# reload the network and look at the layer
model_load_file = dir_data + file_test
print("loading from file {}".format(model_load_file))
file_load = model_load_file
new_model = dcc_basset_lib.load_nasa_model(file_load)

# # print the weights for the index network
# print("layer {} has shape {} and data \n{}".format(index, new_model[index].weight.shape, new_model[index].weight))

