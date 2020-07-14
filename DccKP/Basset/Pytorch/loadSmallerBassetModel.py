# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
# copied from github below for use in my project at work
# https://github.com/kipoi/models/blob/master/Basset/pretrained_model_reloaded_th.py
# see paper at
# http://kipoi.org/models/Basset/


# %%
# imports
import torch
from torch import nn
import twobitreader
from twobitreader import TwoBitFile

print("got pytorch version of {}".format(torch.__version__))


# %%
# import relative libraries
import sys
sys.path.insert(0, '/Users/mduby/Code/WorkspacePython/MachineLearningPython/DccKP/Basset/')
import dcc_basset_lib



# %%
class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))
        


# %%
# load the Basset model
pretrained_model_reloaded_th = nn.Sequential( # Sequential,
        nn.Conv2d(4,300,(19, 1)),
        nn.BatchNorm2d(300),
        nn.ReLU(),
        nn.MaxPool2d((3, 1),(3, 1)),
        nn.Conv2d(300,200,(11, 1)),
        nn.BatchNorm2d(200),
        nn.ReLU(),
        nn.MaxPool2d((4, 1),(4, 1)),
        nn.Conv2d(200,200,(7, 1)),
        nn.BatchNorm2d(200),
        nn.ReLU(),
        nn.MaxPool2d((4, 1),(4, 1)),
        Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2000,1000)), # Linear,
        nn.BatchNorm1d(1000,1e-05,0.1,True),#BatchNorm1d,
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1000,1000)), # Linear,
        nn.BatchNorm1d(1000,1e-05,0.1,True),#BatchNorm1d,
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1000,164)), # Linear,
        nn.Sigmoid(),
    )

print("got model of type {}".format(type(pretrained_model_reloaded_th)))


# %%
# print out the model
print(pretrained_model_reloaded_th)


# %%
# load the weights
# sd = torch.load('/home/javaprog/Data/Broad/Basset/Model/predictions.h5')
sd = torch.load('/Users/mduby/Data/Broad/Basset/Model/pretrained_model_reloaded_th.pth')
pretrained_model_reloaded_th.load_state_dict(sd)



# %%
# summarize the model - LARGE
model_weights = pretrained_model_reloaded_th.state_dict()


# %%
# make the model eval
pretrained_model_reloaded_th.eval()

# better summary
print(pretrained_model_reloaded_th)


# %%
# load the chromosome data
# get the genome file
hg19 = TwoBitFile('/Users/mduby/Data/Broad/Basset/TwoBitReader/hg19.2bit')

print("two bit file of type {}".format(type(hg19)))

# get the chrom
chromosome = hg19['chr11']
position = 95311422

# load the data
ref_sequence, alt_sequence = dcc_basset_lib.get_ref_alt_sequences(position, 300, chromosome, 'C')

print("got ref sequence one hot of type {} and shape {}".format(type(ref_sequence), len(ref_sequence)))
print("got alt sequence one hot of type {} and shape {}".format(type(alt_sequence), len(alt_sequence)))




# %%
# build list and transform into input
sequence_list = []
# sequence_list.append(ref_sequence)
sequence_list.append(ref_sequence)
# sequence_list.append(alt_sequence)
sequence_list.append(alt_sequence)

print(alt_sequence)

# get the np array of right shape
sequence_one_hot = dcc_basset_lib.get_one_hot_sequence_array(sequence_list)
print("got sequence one hot of type {} and shape {}".format(type(sequence_one_hot), sequence_one_hot.shape))


print(sequence_one_hot)

# %%
# create a pytorch tensor
tensor = torch.from_numpy(sequence_one_hot)

print("got pytorch tensor with type {} and shape {} and data type \n{}".format(type(tensor), tensor.shape, tensor.dtype))


# %%
# add a dimension to the tensor and convert to float 32
# FOR SINGLE ELEMENT LIST
# tensor_input = torch.unsqueeze(tensor, 0)
# tensor_input = torch.unsqueeze(tensor_input, 3)

# FRO MULTI
tensor_input = torch.unsqueeze(tensor, 3)
tensor_input = torch.transpose(tensor_input, 1, 2)
tensor_input = tensor_input.to(torch.float)

print("got transposed pytorch tensor with type {} and shape {} and data type \n{}".format(type(tensor_input), tensor_input.shape, tensor_input.dtype))


# %%
# run the model predictions
# pretrained_model_reloaded_th.eval()
predictions = pretrained_model_reloaded_th(tensor_input)

print("got predictions of type {} and shape {} and result \n{}".format(type(predictions), predictions.shape, predictions))


print("got 0,1 prediction {}".format((predictions[0,2] - predictions[1,2]).item()))

# get the absolute value of the difference
tensor_abs = torch.abs(predictions[0] - predictions[1])

print(tensor_abs)

# open the label file
with open('/Users/mduby/Data/Broad/Basset/Model/labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]

print("the labels of type {} and length {} are \n{}".format(type(labels), len(labels), labels))

result_map = {}
for index in range(0, 164):
    result_map[labels[index]] = tensor_abs[index].item()

print("the result of type {} and length {} are \n{}".format(type(result_map), len(result_map), result_map))





