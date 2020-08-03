# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
# copied from github below for use in my project at work
# https://github.com/kipoi/models/blob/master/Basset/pretrained_model_reloaded_th.py
# see paper at
# http://kipoi.org/models/Basset/


# imports
import torch
from torch import nn
import twobitreader
from twobitreader import TwoBitFile
import time

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
# file_model_weights = dir_data + 'Basset/Model/dude_model.pth'
# file_model_weights = dir_data + 'Basset/Model/pretrained_model_reloaded_th.pth'
file_model_weights = dir_data + 'Basset/Production/basset_pretrained_model_reloaded.pth'
file_twobit = dir_data + 'Basset/Production/hg19.2bit'
labels_file = dir_data + '/Basset/Production/basset_labels.txt'

# open the label file
with open(labels_file) as f:
    labels_list = [line.strip() for line in f.readlines()]

# load the chromosome data
# get the genome file
hg19 = TwoBitFile(file_twobit)
print("two bit file of type {}".format(type(hg19)))

# LOAD THE MODEL
# load the weights
pretrained_model_reloaded_th = dcc_basset_lib.load_basset_model(file_model_weights)
# pretrained_model_reloaded_th = dcc_basset_lib.load_nasa_model(file_model_weights)

# make the model eval
pretrained_model_reloaded_th.eval()

# better summary
print(pretrained_model_reloaded_th)

# LOAD THE INPUTS
# load the list of variants
variant_list = dcc_basset_lib.get_variant_list(file_input)

print("got variant list of size {}".format(len(variant_list)))

# split into chunks
chunk_size = 1000 # 20s, 153 chunks - so 50 mins per file, 200 x 50 = 10,000 mins on PC
# chunk_size = 2000
chunks = [variant_list[x:x+chunk_size] for x in range(0, len(variant_list), chunk_size)]
print("got chunk list of size {} and type {}".format(len(chunks), type(chunks)))

# loop through chunks
main_start_time = time.perf_counter()
final_results = []
for chunk_index in range(0, len(chunks)):
# for chunk_index in range(6, 7):
    variant_list = chunks[chunk_index]

    # get start time
    start_time = time.perf_counter()

    # get the sequence input for the first chunk
    # variant_list = chunks[0]
    variant_list, tensor_input = dcc_basset_lib.get_input_tensor_from_variant_list(variant_list, hg19, 600, False)

    # get end time
    end_time = time.perf_counter()
    print("({}) generated input tensor of shape {} in {:0.4}s".format(chunk_index, tensor_input.shape, end_time - start_time))

    # get start time
    start_time = time.perf_counter()

    # run the model predictions
    pretrained_model_reloaded_th.eval()
    predictions = pretrained_model_reloaded_th(tensor_input)

    # get end time
    end_time = time.perf_counter()
    print("generated predictions tensor of shape {} in {:0.4}s".format(predictions.shape, end_time - start_time))

    # get start time
    start_time = time.perf_counter()

    # get the result map
    result_list = dcc_basset_lib.get_result_map(variant_list, predictions, labels_list)
    final_results.extend(result_list)
    # print("got result list {}".format(result_list))

    # get end time
    end_time = time.perf_counter()
    print("got result list of size {} in time {:0.4f}s".format(len(result_list), end_time - start_time))

# end
main_end_time = time.perf_counter()
print("got final results of size {} in time {:0.4f}".format(len(final_results), main_end_time - main_start_time))



