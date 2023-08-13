

# import
import os 
import xml.etree.ElementTree as ET
import requests
import io
import json
import xml
import time


# import relative libraries
dir_code = "/home/javaprog/Code/PythonWorkspace/"
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/GPT/')
import dcc_gpt_lib

# constants
SCHEMA_GPT = "gene_gpt"
DB_PAPER_TABLE = "pgpt_paper"
DB_PAPER_ABSTRACT = "pgpt_paper_abtract"


# methods
def create_list_from_string(list_str, log=False):
    '''
    will create a list from the combination of comma seperated string
    '''
    # intialize
    list_result = []

    # loop
    for row in list_str:
        # split stribng
        list_temp = row.split(",")

        # log
        if log:
            print("adding list of size: {} to list of size: {}".format(len(list_temp), len(list_result)))

        # add
        list_result = list_result + list_temp

    # return
    return list_result




# main
if __name__ == "__main__":
    # get the connection
    conn = dcc_gpt_lib.get_connection(schema=SCHEMA_GPT)

    # get the list of genes
    list_genes = create_list_from_string(dcc_gpt_lib.LIST_MASTER, log=True)
    print("got list: {}".format(list_genes))

    # create searches
    for gene in list_genes:
        dcc_gpt_lib.insert_db_search(conn=conn, gene=gene, to_dowwnload='Y', to_download_ids='Y', log=True)

    for gene in list_genes:
        dcc_gpt_lib.update_db_search_to_download_by_gene(conn=conn, gene=gene, to_download='Y')

