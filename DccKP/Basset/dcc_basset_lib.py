# imports
import twobitreader
from twobitreader import TwoBitFile
import numpy as np 
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder
import csv

print("have pytorch version {}".format(torch.__version__))
print("have numpy version {}".format(np.__version__))

# add in model classes
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


# method to create string sequence from position
def get_genomic_sequence(position, offset, chromosome, alt_allele=None):
    if alt_allele is not None:
        sequence = chromosome[position - offset : position - 1] + alt_allele + chromosome[position : position + offset]
    else:
        sequence = chromosome[position - offset: position + offset]

    # return
    return sequence.upper()

def get_ref_alt_sequences(position, offset, chromosome, alt_allele=None):
    ref_sequence = get_genomic_sequence(position, offset, chromosome)
    alt_sequence = get_genomic_sequence(position, offset, chromosome, alt_allele)

    return ref_sequence, alt_sequence

def get_input_np_array(sequence_list):
    sequence_np = None
    for seq in sequence_list:
        if sequence_np is None:
            sequence_np = np.array(list(seq))
        else:
            sequence_np = np.vstack((sequence_np, np.array(list(seq))))

    # return
    return sequence_np

def get_one_hot_sequence_array(sequence_list):
    # get the numpy sequence
    sequence_np = get_input_np_array(sequence_list)

    # use the numpy utility to replace the letters by numbers
    sequence_np[sequence_np == 'A'] = 0
    sequence_np[sequence_np == 'C'] = 1
    sequence_np[sequence_np == 'G'] = 2
    sequence_np[sequence_np == 'T'] = 3

    # convert to ints
    sequence_np = sequence_np.astype(np.int)

    # one hot the sequence
    number_classes = 4
    sequence_np = np.eye(number_classes)[sequence_np]

    # return
    return sequence_np

def get_variant_list(file):
    variants = []
    with open(file, 'r') as variant_file:
        cvsreader = csv.reader(variant_file)

        # skip the header
        next(cvsreader)

        # read all the next rows
        for row in cvsreader:
            variants.append(row[0].rstrip().split('\t')[0])


    # print the first 10 variants
    for index in range(1, 10):
        print("got variant: {}".format(variants[index]))

    # return
    return variants

def load_basset_model(weights_file, should_log=True):
    # load the weights
    state_dict = torch.load(weights_file)
    print("dude \n{}\n\n".format(state_dict))
    pretrained_model_reloaded_th = load_basset_model_from_state_dict(state_dict, should_log)

    # return
    return pretrained_model_reloaded_th

def load_nasa_model(weights_file, should_log=True):
    # load the weights
    state_dict = torch.load(weights_file)
    print("dude \n{}\n\n".format(state_dict))
    pretrained_model_reloaded_th = load_nasa_model_from_state_dict(state_dict, should_log)

    # return
    return pretrained_model_reloaded_th

def load_basset_model_from_state_dict(state_dict, should_log=True):
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

    # print
    if should_log:
        print("got model of type {}".format(type(pretrained_model_reloaded_th)))

    # load the weights
    pretrained_model_reloaded_th.load_state_dict(state_dict)

    # return
    return pretrained_model_reloaded_th

def load_nasa_model_from_state_dict(state_dict, should_log=True):
    # load the Basset model
    pretrained_model_reloaded_th = nn.Sequential( # Sequential,
        nn.Conv2d(4,400,(21, 1),(1, 1),(10, 0)),
        nn.BatchNorm2d(400),
        nn.ReLU(),
        nn.MaxPool2d((3, 1),(3, 1),(0, 0),ceil_mode=True),
        nn.Conv2d(400,300,(11, 1),(1, 1),(5, 0)),
        nn.BatchNorm2d(300),
        nn.ReLU(),
        nn.MaxPool2d((4, 1),(4, 1),(0, 0),ceil_mode=True),
        nn.Conv2d(300,300,(7, 1),(1, 1),(3, 0)),
        nn.BatchNorm2d(300),
        nn.ReLU(),
        nn.MaxPool2d((4, 1),(4, 1),(0, 0),ceil_mode=True),
        nn.Conv2d(300,300,(5, 1),(1, 1),(2, 0)),
        nn.BatchNorm2d(300),
        nn.ReLU(),
        nn.MaxPool2d((4, 1),(4, 1),(0, 0),ceil_mode=True),
        Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1500,1024)), # Linear,
        nn.BatchNorm1d(1024,1e-05,0.1,True),#BatchNorm1d,
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1024,512)), # Linear,
        nn.BatchNorm1d(512,1e-05,0.1,True),#BatchNorm1d,
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(512,167)), # Linear,
        nn.Sigmoid(),
    )
        # pretrained_model_reloaded_th = nn.Sequential( # Sequential,
        #     nn.Conv2d(4,400,(21, 1)),
        #     nn.BatchNorm2d(400),
        #     nn.ReLU(),
        #     nn.MaxPool2d((3, 1),(3, 1)),
        #     nn.Conv2d(400,300,(11, 1)),
        #     nn.BatchNorm2d(300),
        #     nn.ReLU(),
        #     nn.MaxPool2d((4, 1),(4, 1)),
        #     nn.Conv2d(300,300,(7, 1)),
        #     nn.BatchNorm2d(300),
        #     nn.ReLU(),
        #     nn.MaxPool2d((4, 1),(4, 1)),
        #     nn.Conv2d(300,300,(5, 1)),
        #     nn.BatchNorm2d(200),
        #     nn.ReLU(),
        #     nn.MaxPool2d((4, 1),(4, 1)),
        #     Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
        #     nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1500,1024)), # Linear,
        #     nn.BatchNorm1d(1024,1e-05,0.1,True),#BatchNorm1d,
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1024,512)), # Linear,
        #     nn.BatchNorm1d(1000,1e-05,0.1,True),#BatchNorm1d,
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(512,167)), # Linear,
        #     nn.Sigmoid(),
        # )

    # print
    if should_log:
        print("got model of type {}".format(type(pretrained_model_reloaded_th)))

    # load the weights
    if state_dict is not None:
        pretrained_model_reloaded_th.load_state_dict(state_dict)

    # return
    return pretrained_model_reloaded_th

def generate_input_tensor(variant_list, file_twobit):
    # load the chromosome data
    # get the genome file
    hg19 = TwoBitFile(file_twobit)

    # loop for through the variants
    # for variant in variant_list:

    # get the chrom
    chromosome = hg19['chr11']
    position = 95311422

    # load the data
    ref_sequence, alt_sequence = dcc_basset_lib.get_ref_alt_sequences(position, 300, chromosome, 'C')

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

def split_variant(variant):
    pieces = variant.split(":")
    return 




if __name__ == '__main__':
    # set the data dir
    dir_data = "/Users/mduby/Data/Broad/"
    file_input = dir_data + "Magma/Common/part-00011-6a21a67f-59b3-4792-b9b2-7f99deea6b5a-c000.csv"
#    file_model_weights = dir_data + 'Basset/Model/pretrained_model_reloaded_th.pth'
    file_model_weights = dir_data + 'Basset/Nasa/ampt2d_cnn_900_check.th'
    file_twobit = dir_data + 'Basset/TwoBitReader/hg19.2bit'


    # get the genome file
    hg19 = TwoBitFile(dir_data + 'Basset/TwoBitReader/hg19.2bit')

    print("two bit file of type {}".format(type(hg19)))

    # get the chrom
    chromosome = hg19['chr11']
    position = 95311422
    # chromosome = hg19['chr8']
    # position = 118184783

    print("two bit chromosome of type {}".format(type(chromosome)))

    # get the regular sequence
    ref_sequence = get_genomic_sequence(position, 3, chromosome)
    print("got ref sequence: {}".format(ref_sequence))

    # get the allele sequence
    alt_sequence = get_genomic_sequence(position, 3, chromosome, 'C')
    print("got alt sequence: {}".format(alt_sequence))
    print()

    # get ref and alt sequences
    ref_sequence, alt_sequence = get_ref_alt_sequences(position, 3, chromosome, 'C')
    print("got ref sequence: {}".format(ref_sequence))
    print("got alt sequence: {}".format(alt_sequence))
    print()

    sequence_list = []
    sequence_list.append(ref_sequence)
    sequence_list.append(alt_sequence)
    print("input sequence list {}".format(sequence_list))
    print()

    sequence_numpy = get_input_np_array(sequence_list)
    print("got sequence input of type {} and shape {} of\n{}".format(type(sequence_numpy), sequence_numpy.shape, sequence_numpy))
    print()

    sequence_one_hot = get_one_hot_sequence_array(sequence_list)
    print("got sequence one hot of type {} and shape {} of\n{}".format(type(sequence_one_hot), sequence_one_hot.shape, sequence_one_hot))
    print()

    # read the variant file
    variant_file = dir_data + 'Magma/Common/part-00011-6a21a67f-59b3-4792-b9b2-7f99deea6b5a-c000.csv'
    variant_list = get_variant_list(variant_file)
    for index in range(1, 10):
        print("got variant: {}".format(variant_list[index]))

    # load the pytorch model
    pretrained_model_reloaded_th = load_basset_model(file_model_weights)
    pretrained_model_reloaded_th.eval()

    # better summary of the model
    print(pretrained_model_reloaded_th)

    # split a variant
    variant_id = "1:65359821:G:A"
    variant_pieces = split_variant(variant_id)
    print("got variant pieces: {}".format(variant_pieces))

