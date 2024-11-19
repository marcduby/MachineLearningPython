

# Read the input file and write to a new output file without the second column
DIR_LOCATION = '/home/javaprog/Data/Broad/Translator/BayesGenSetNMF/GeneListsRummaGene/'
file_input = '{}/rummagene.txt'.format(DIR_LOCATION)
file_output = '{}/rummageneNoSupplement.txt'.format(DIR_LOCATION)


if __name__ == "__main__":
    with open(file_input, 'r') as infile, open(file_output, 'w') as outfile:
        for line in infile:
            columns = line.strip().split('\t')  # Split the line into columns
            columns.pop(1)  # Remove the second column (index 1)
            outfile.write('\t'.join(columns) + '\n')  # Write the modified line to the output file

