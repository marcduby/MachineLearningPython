# imports
import csv

# read the file
variants = []
file = '/Users/mduby/Data/Broad/Magma/Common/part-00011-6a21a67f-59b3-4792-b9b2-7f99deea6b5a-c000.csv'
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


