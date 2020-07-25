# imports
import torch
from torch import nn

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


# build network by class
class SequentialGenetic(nn.Sequential):
    def __init__(self):
        super(SequentialGenetic, self).__init__()
        self.conv01 = nn.Conv2d(4,400,(21, 1),(1, 1),(10, 0))
        self.batchnorm01 = nn.BatchNorm2d(400)
        self.relu01 = nn.ReLU()
        self.maxpool01 = nn.MaxPool2d((3, 1),(3, 1),(0, 0),ceil_mode=True)
        self.conv02 = nn.Conv2d(400,300,(11, 1),(1, 1),(5, 0))
        self.batchnorm02 = nn.BatchNorm2d(300)
        self.relu02 = nn.ReLU()
        self.maxpool02 = nn.MaxPool2d((4, 1),(4, 1),(0, 0),ceil_mode=True)

    def forward(self, t):
        return t

# create new model and print
seq_model = SequentialGenetic()
print(seq_model)

# print a layer of the network
layer = seq_model.conv02
print(layer)

# print the layer weight
print("the layer weights shape is {}".format(layer.weight.shape))
print("the layer bias shape is {}".format(layer.bias.shape))




