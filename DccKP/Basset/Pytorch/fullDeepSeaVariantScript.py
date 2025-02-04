# implementation of the DeepSEA variant effect PyTorch model detailed below:
# http://kipoi.org/models/DeepSEA/variantEffects/
# 
# see paper: https://www.nature.com/articles/nmeth.3547

# imports
import torch
from torch import nn
import twobitreader
from twobitreader import TwoBitFile
import time
import argparse
import json

print("got pytorch version of {}".format(torch.__version__))

# set the code and data directories
# dir_code = ""
# dir_data = ""
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/home/javaprog/Data/Broad/DeepSEA/Models/"

# import relative libraries
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Basset/')
import dcc_basset_lib

# file input
file_input = dir_data + "../../dig-analysis-data/out/varianteffect/common/part-00011-6a21a67f-59b3-4792-b9b2-7f99deea6b5a-c000.csv"
file_twobit = dir_data + 'hg19.2bit'
file_output = dir_data + "../../dig-analysis-data/out/regionpytorch/deepsea/basset_part-00011-6a21a67f-59b3-4792-b9b2-7f99deea6b5a-c000.csv"
file_model_weights = dir_data + 'deepsea_cpu.pth'
labels_file = dir_data + 'deepsea_labels.txt'

# set the region size
region_size = 1000       # deepsea variant
# region_size = 600       # basset
# region_size = 900       # nasa

# chunk_size = 1000 # 20s, 153 chunks - so 50 mins per file, 200 x 50 = 10,000 mins on PC
batch_size = 20

# read in the passed in file if any
# configure argparser
parser = argparse.ArgumentParser("script to run the deepsea variant effect ML model in the aggregator")
# add the arguments
parser.add_argument('-i', '--input_file', help='the file to process', default=file_input, required=True)
parser.add_argument('-o', '--output_file', help='the file to save the results to', default=file_output, required=True)
parser.add_argument('-b', '--batch', help='the batch size to process', default=batch_size, required=False)
# get the args
args = vars(parser.parse_args())
if args['input_file'] is not None:
    file_input = args['input_file']
if args['output_file'] is not None:
    file_output = args['output_file']
if args['batch'] is not None:
    batch_size = int(args['batch'])
print("using variant input file {} and resule output file {} with batch size {}".format(file_input, file_output, batch_size))

# open the label file
with open(labels_file) as f:
    labels_list = [line.strip() for line in f.readlines()]

# load the chromosome data
# get the genome file
hg19 = TwoBitFile(file_twobit)
print("two bit file of type {}".format(type(hg19)))

# LOAD THE MODEL
# load the weights
pretrained_model_reloaded_th = dcc_basset_lib.load_deepsea_varianteffect_model(file_model_weights)

# make the model eval
pretrained_model_reloaded_th.eval()

# better summary
print(pretrained_model_reloaded_th)

# LOAD THE INPUTS
# load the list of variants
variant_list = dcc_basset_lib.get_variant_list(file_input)
print("got variant list of size {}".format(len(variant_list)))

# split into chunks
# chunk_size = 2000
chunks = [variant_list[x : x+batch_size] for x in range(0, len(variant_list), batch_size)]
print("got chunk list of size {} and type {}".format(len(chunks), type(chunks)))

# loop through chunks
main_start_time = time.perf_counter()
final_results = []
# for chunk_index in range(0, len(chunks)):
for chunk_index in range(6, 9):
    variant_list = chunks[chunk_index]

    # get start time
    start_time = time.perf_counter()

    # get the sequence input for the first chunk
    # variant_list = chunks[0]
    variant_list, tensor_input = dcc_basset_lib.get_input_tensor_from_variant_list(variant_list, hg19, region_size, False)

    # get end time
    end_time = time.perf_counter()
    print("({}) generated input tensor of shape {} in {:0.4}s".format(chunk_index, tensor_input.shape, end_time - start_time))

    # get start time
    start_time = time.perf_counter()

    # NOTE: permute the tensor for the deepsea vs variant inputs
    # tensor input is (batch, 4 nucleotides, 1 channel, 1000bp region size)
    tensor_input = tensor_input.permute(0, 1, 3, 2)

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

# write out the output
with open(file_output, 'w') as out_file:
    out = json.dumps(final_results)
    out_file.write(out)
    print("got final results file {}".format(file_output))


