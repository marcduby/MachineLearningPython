

# import
import os 
import xml.etree.ElementTree as ET
import requests
import io
import json
import xml
import time

# for AWS
ENV_DIR_CODE = os.environ.get('DIR_CODE')
ENV_DIR_PUBMED = os.environ.get('DIR_PUBMED')

# import relative libraries
dir_code = "/home/javaprog/Code/PythonWorkspace/"
if ENV_DIR_CODE:
    dir_code = ENV_DIR_CODE
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/GPT/')
import dcc_gpt_lib

# constants
SCHEMA_GPT = "gene_gpt"
DB_PAPER_TABLE = "pgpt_paper"
DB_PAPER_ABSTRACT = "pgpt_paper_abtract"


# methods

# main
if __name__ == "__main__":
    # get the connection
    conn = dcc_gpt_lib.get_connection(schema=SCHEMA_GPT)

    # get the list of genes
    list_genes = dcc_gpt_lib.create_list_from_string_list(dcc_gpt_lib.LIST_MASTER, log=True)
    print("got list: {}".format(list_genes))

    # create searches
    for gene in list_genes:
        dcc_gpt_lib.insert_db_search(conn=conn, gene=gene, to_dowwnload='Y', to_download_ids='Y', log=True)

    # update for download
    for gene in list_genes:
        dcc_gpt_lib.update_db_search_to_download_by_gene(conn=conn, gene=gene, to_download='Y')

    # update for ready to summarize
    for gene in list_genes:
        dcc_gpt_lib.update_db_search_ready_by_gene(conn=conn, gene=gene, ready='Y')

