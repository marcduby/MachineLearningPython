

# imports
import pandas as pd 
import pymysql as mdb
import requests 
import numpy as np
import os 
import time
import logging
import sys


# import relative libraries
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/home/javaprog/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl


# constants
sql_get_null_name_curies = "select distinct gb.phenotype_ontology_id from data_genebass_gene_phenotype_good_prob gb where gb.phenotype_genepro_name is null"
sql_update_null_name_curies = "update data_genebass_gene_phenotype_good_prob gb set gb.phenotype_genepro_name = %s where gb.phenotype_genepro_name is null and gb.phenotype_ontology_id = %s"
sql_get_null_gene_curies = "select distinct gb.gene from data_genebass_gene_phenotype_good_prob gb where gb.gene_ncbi_id is null"
sql_update_null_gene_curies = "update data_genebass_gene_phenotype_good_prob gb set gb.gene_ncbi_id = %s where gb.gene_ncbi_id is null and gb.gene = %s"

bool_get_name = False
bool_update_phenotypes = False
bool_get_genes = True
bool_update_genes = True
url_node_normalizer = "https://nodenormalization-sri.renci.org/1.1/get_normalized_nodes?curie={}"
DB_PASSWD = os.environ.get('DB_PASSWD')
logging.basicConfig(level=logging.INFO, format=f'[%(asctime)s] - %(levelname)s - %(name)s : %(message)s')
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)

# methods
def get_normalizer_data_name(curie_id, debug=True):
    ''' calls the node normlizer and returns the name and asked for curie id '''
    result_name = None
    url = url_node_normalizer.format(curie_id)

    # log
    if debug:
        print("looking up curie: {}".format(curie_id))
        print("looking up url: {}".format(url))

    # call the normalizer
    response = requests.get(url)
    json_response = response.json()
    if debug:
        logger.info(json_response)

    # get the data from the response
    try:
        if json_response:
            result_name = json_response.get(curie_id).get("id").get("label")
        else:
            logger.error("ERROR: got no response for curie {}".format(curie_id))
    except:
        logger.error("ERROR: got no response for curie {}".format(curie_id))

    # log
    if debug:
        logger.info("got name: {}, curie id: {}".format(result_name, curie_id))

    # return
    return result_name

# main
if __name__ == "__main__":
    counter = 0

    # get the db connection
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db='tran_dataload')
    cursor = conn.cursor()

    # update the table
    if bool_get_name:
        # get the curies with missing names
        cursor.execute(sql_get_null_name_curies)
        results = cursor.fetchall()

        # get the names from the node normalizer
        for row in results:
            counter += 1
            curie_id = row[0]
            curie_name = get_normalizer_data_name(curie_id, debug=False)

            # print
            logger.info("{} - for {} got name '{}'".format(counter, curie_id, curie_name))

            # update if not null
            if bool_update_phenotypes:
                if curie_name:
                    cursor.execute(sql_update_null_name_curies, (curie_name, curie_id)) 
                    logger.info("updated {} to name '{}'".format(curie_id, curie_name))

            # sleep
            logger.info("-----------------")
            time.sleep(1)

            # commit every 10
            if counter % 10 == 0:
                # commit
                conn.commit()
                logger.info("{} - committed".format(counter))

    # update the table gene data
    if bool_get_genes:
        # get the genes with missing curies
        cursor.execute(sql_get_null_gene_curies)
        results = cursor.fetchall()

        # get the names from the node normalizer
        for row in results:
            counter += 1
            gene = row[0]
            ontology_id = tl.find_ontology(gene, ['NCBIGene'], debug=True)

            # print
            logger.info("{} - for {} got curie '{}'".format(counter, gene, ontology_id))

            # update if not null
            if bool_update_genes:
                if ontology_id:
                    cursor.execute(sql_update_null_gene_curies, (ontology_id, gene)) 
                    logger.info("updated {} to curie '{}'".format(gene, ontology_id))

            # sleep
            logger.info("-----------------")
            time.sleep(1)

            # commit every 10
            if counter % 10 == 0:
                # commit
                conn.commit()
                logger.info("{} - committed".format(counter))

