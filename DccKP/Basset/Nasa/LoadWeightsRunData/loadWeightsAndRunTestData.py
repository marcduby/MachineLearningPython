# imports
import torch
from torch import nn
import twobitreader
from twobitreader import TwoBitFile
from torch.utils.serialization import load_lua

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
file_model_weights = dir_data + 'Basset/Nasa/ampt2d_cnn_900_best_cpu.th'
file_twobit = dir_data + 'Basset/TwoBitReader/hg19.2bit'

# LOAD THE MODEL
# load the weights
state_dict = load_lua(file_model_weights)

# get the models
lua_model = state_dict.model
nasa_model = dcc_basset_lib.load_nasa_model_from_state_dict(None)

# count = 0
# print("model.modules is of type {}".format(type(model.modules)))
# for module in model.modules:
#     print("({}) model name {} and type {} and weights {}".format(count, module, type(module), module.bias.shape))
#     count = count + 1

# for i in (0, 1):
#     module = model.modules[i]
#     print("({}) model name {} and type {} and weights {}".format(i, module, type(module), module.weight.shape))

# print layer weight before setting
# index = 1
# module = nasa_model[index]
# print("layer {} has shape {} and weights {}".format(index, module.weight.shape, module.weight))
# module.weight = nn.Parameter(model.modules[index].weight)
# module.bias = nn.Parameter(model.modules[index].bias)
# print("layer {} has shape {} and weights {}".format(index, module.weight.shape, module.weight))

# need to transpose dimensions 2 and 3 for the con2d layers
for index in (0, 4, 8, 12):
    old_module = lua_model.modules[index]
    # print("({}) model name {} and type {} and weights {}".format(index, old_module, type(old_module), old_module.weight.shape))
    nasa_module = nasa_model[index]
    print("({}) setting params for layer name {} has type {} and weights {}".format(index, nasa_module, type(nasa_module), nasa_module.weight.shape))
    with torch.no_grad():
        nasa_module.weight = nn.Parameter(torch.transpose(lua_model.modules[index].weight.type(torch.FloatTensor), 2, 3))
        nasa_module.bias = nn.Parameter(lua_model.modules[index].bias.type(torch.FloatTensor))

# layer that do not need transposition
for index in (1, 5, 9, 13, 18, 22):
    old_module = lua_model.modules[index]
    # print("({}) model name {} and type {} and weights {}".format(index, old_module, type(old_module), old_module.weight.shape))
    nasa_module = nasa_model[index]
    print("({}) setting params for layer name {} has type {} and weights {}".format(index, nasa_module, type(nasa_module), nasa_module.weight.shape))
    with torch.no_grad():
        nasa_module.weight = nn.Parameter(lua_model.modules[index].weight.type(torch.FloatTensor))
        nasa_module.bias = nn.Parameter(lua_model.modules[index].bias.type(torch.FloatTensor))

# linear sub layers
for index in (17, 21, 25):
    old_module = lua_model.modules[index]
    # print("({}) model name {} and type {} and weights {}".format(index, old_module, type(old_module), old_module.weight.shape))
    nasa_module = nasa_model[index][1]
    print("({}) setting params for layer name {} has type {} and weights {}".format(index, nasa_module, type(nasa_module), nasa_module.weight.shape))
    with torch.no_grad():
        nasa_module.weight = nn.Parameter(lua_model.modules[index].weight.type(torch.FloatTensor))
        nasa_module.bias = nn.Parameter(lua_model.modules[index].bias.type(torch.FloatTensor))

# evaluate the network
nasa_model.eval()

# better summary
print(nasa_model)


# LOAD THE INPUTS
# load the list of variants
variant_list = dcc_basset_lib.get_variant_list(file_input)

print("got variant list of size {}".format(len(variant_list)))

# split into chunks
chunk_size = 100
chunks = [variant_list[x:x+chunk_size] for x in range(0, len(variant_list), chunk_size)]
print("got chunk list of size {} and type {}".format(len(chunks), type(chunks)))

print("got chunks data {}".format(chunks[0][0]))

# load the chromosome data
# get the genome file
hg19 = TwoBitFile(file_twobit)

print("two bit file of type {}".format(type(hg19)))

# get the chrom
# chromosome = hg19['chr11']
# position = 95311422
chromosome = hg19['chr17']
position = 65867911

# load the data
ref_sequence, alt_sequence = dcc_basset_lib.get_ref_alt_sequences(position, 450, chromosome, 'C')

print("got ref sequence one hot of type {} and shape {}".format(type(ref_sequence), len(ref_sequence)))
print("got alt sequence one hot of type {} and shape {}".format(type(alt_sequence), len(alt_sequence)))

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
# print(sequence_one_hot)

# create a pytorch tensor
tensor = torch.from_numpy(sequence_one_hot)

print("got pytorch tensor with type {} and shape {} and data type \n{}".format(type(tensor), tensor.shape, tensor.dtype))

# build the input tensor
tensor_initial = torch.unsqueeze(tensor, 3)
tensor_input = tensor_initial.permute(0, 2, 1, 3)
tensor_input = tensor_input.to(torch.float)

print("got transposed pytorch tensor with type {} and shape {} and data type \n{}".format(type(tensor_input), tensor_input.shape, tensor_input.dtype))

# run the model predictions
# pretrained_model_reloaded_th.eval()
predictions = nasa_model(tensor_input)

print("got predictions of type {} and shape {} and result \n{}".format(type(predictions), predictions.shape, predictions))
# print("got 0,1 prediction {}".format((predictions[0,2] - predictions[1,2]).item()))

# get the absolute value of the difference
# tensor_abs = torch.abs(predictions[0] - predictions[1])
tensor_abs = torch.abs(predictions[0])
print(tensor_abs)

# open the label file
with open(dir_data + '/Basset/Model/labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# print("the labels of type {} and length {} are \n{}".format(type(labels), len(labels), labels))

result_map = {}
for index in range(0, 164):
    result_map[labels[index]] = tensor_abs[index].item()

print("the result of type {} and length {} are \n{}".format(type(result_map), len(result_map), result_map))

