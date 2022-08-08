# imports
import argparse
import os
from glob import glob
import json
import csv

# input and output directories
dir_s3 = f'/home/javaprog/Data/Broad/dig-analysis-data/'
# dir_s3 = f's3://dig-analysis-data'
dir_pathway = f'{dir_s3}/bin/magma/pathwayFiles/'
file_ncbi_gene = f'{dir_s3}/bin/magma/NCBI37.3.gene.loc'
file_out = f'{dir_s3}/out/magma/pathway-genes/pathwayGenes.txt'
debug = True
count_debug = 5

def main():
    """
    Arguments: None
    """
    # initialize
    map_pathway_genes = {}
    map_gene_id = {}

    # read the gene file
    with open(file_ncbi_gene) as csv_file:
        for line in csv_file:
            fields = line.split('\t')
            map_gene_id[fields[5].strip()] = fields[0].strip()

    # log
    if debug:
        count = 0
        for key, value in map_gene_id.items():
            count = count+1
            print("{} - {}".format(key, value))
            if count > count_debug:
                break

    # load all the files in the directory
    for filename in glob(dir_pathway + "*.json"):
        with open(filename) as json_file:
            json_data = json.load(json_file)

            # loop through the data
            count = 0
            for key, value in json_data.items():
                list_genes = []

                # debug
                if debug:
                    count = count + 1
                    if count > 2000000:
                        break

                # get the pathway name
                pathway_name = key

                # add the list of genes for the pathway
                pathway_genes = value.get('geneSymbols')

                # for each gene, translate to the ncbi gene id
                for gene in pathway_genes:
                    gene_id = map_gene_id.get(gene)

                    if gene_id:
                        list_genes.append(gene_id)

                # store in the map
                map_pathway_genes[pathway_name] = " ".join(list_genes)

    # log
    count = 0
    for key, value in map_pathway_genes.items():
        count = count+1
        print("{} - {}".format(key, value))
        if count > count_debug:
            break

    # delete file if exists 
    try:
        os.remove(file_out)
    except OSError:
        pass

    # write to file
    with open(file_out, "a") as file_result:
        for key, value in map_pathway_genes.items():
            if len(value) > 0:
                file_result.write("{} {}\n".format(key, value))

if __name__ == '__main__':
    main()
