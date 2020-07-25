# import
import torch
from torch import nn

print("the pytorch version is {}".format(torch.__version__))



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

