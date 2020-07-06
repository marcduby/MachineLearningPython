# copied from github below for use in my project at work
# https://github.com/kipoi/models/blob/master/Basset/pretrained_model_reloaded_th.py
# see paper at
# http://kipoi.org/models/Basset/

# imports
import torch
from torch import nn
import twobitreader
from twobitreader import TwoBitFile

print("got pytorch version of {}".format(torch.__version__))


# load the model
model = torch.load('/home/javaprog/Data/Broad/Basset/Model/pretrained_model.th')
print(model)

