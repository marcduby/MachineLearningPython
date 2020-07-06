# imports
import twobitreader
from twobitreader import TwoBitFile
import numpy as np 
import torch
from sklearn.preprocessing import OneHotEncoder

print("have pytorch version {}".format(torch.__version__))
print("have numpy version {}".format(np.__version__))

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


if __name__ == '__main__':
    # get the genome file
    hg19 = TwoBitFile('../../../Data/Broad/Basset/TwoBitReader/hg19.2bit')

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


