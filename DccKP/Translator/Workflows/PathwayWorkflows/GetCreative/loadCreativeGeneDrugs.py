
# imports
# import pandas as pd 
import pymysql as mdb
import requests 
# import numpy as np
import os 
import json
import sys

# constants
dir_data = "/Users/mduby/Data/Broad/"
dir_data = "/home/javaprog/Data/Broad/"
dir_code = "/home/javaprog/Code/PythonWorkspace/"
file_pathways = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/GoogleDistancePathways/pathwayInformation.json"
is_insert_data = True
is_update_data = True
DB_PASSWD = os.environ.get('DB_PASSWD')
location_input_query = dir_code + "MachineLearningPython/DccKP/Translator/Workflows/Json/Queries/GetCreative/drugGeneQuery.json"
max_count = 5000
url_molepro = "https://translator.broadinstitute.org/molepro/trapi/v1.2/query"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl


# sql statements
sql_select = """ select id, gene_node_id, gene_ontology_id, gene_code from tran_upkeep.molepro_gene_status where load_satus != 'done' """
sel_update_status = """ update tran_upkeep.molepro_gene_status set load_status = 'done' where id = ? """
sql_insert = """ insert into tran_upkeep.molepro_drug_gene 
    (gene_node_id, gene_ontology_id, gene_code, drug_ontology_id, drug_name, drug_categore_biolink_id, predicate_biolink_id) 
    values(%s, %s, %s, %s, %s, %s, %s) """

# get the query
with open(location_input_query) as file_json: 
    json_trapi_query = json.load(file_json)

# functions
def get_result_list(json, gene_id, log=False):
    '''
    parse the json trapi result
    '''
    list_result = []

    # get the nodes and edges map
    map_nodes = json.get('message').get('knowledge_graph').get('nodes')
    map_edges = json.get('message').get('knowledge_graph').get('edges')

    # log empyt
    if len(map_nodes) < 2:
        print("no nodes")
    elif len(map_edges) < 1:
        print("no edges")
    else:
        print(map_edges)
        for key, value in map_edges.items():
            drug_id = value.get('subject')
            drug_name = map_nodes.get(drug_id).get('name')
            print("got drug: {} - {}".format(drug_id, drug_name))
            list_result.append({'gene_id': gene_id, 'drug_id': drug_id, 
                'predicate': value.get('predicate'), 'drug_name': drug_name})

    # return
    return list_result


# main
if __name__ == "__main__":
    # initialize
    counter = 0

    # connect to the database
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db='tran_upkeep')
    cursor = conn.cursor()

    # get the gene rows
    cursor.execute(sql_select)
    db_results = cursor.fetchall()

    # loop for each gene
    for item in db_results:
        counter = counter + 1
        row_id, gene_id, gene_ontology_id, gene_code = item[0], item[1], item[2], item[3]

        # put gene id in query
        print("{} - querying for gene: {}".format(counter, gene_id))
        json_trapi_query.get('message').get('query_graph').get('nodes').get('gene')['ids'] = [gene_id]

        # for each gene, query
        response = requests.post(url_molepro, json=json_trapi_query)

        # try and catch exception
        try:
            json_output = response.json()
            # print("got result: \n{}".format(json_output))
        except ValueError:
            print("GOT ERROR: skipping for gene: {}".format(gene_id))
            continue

        # get the drug list
        list_drugs = get_result_list(json_output, gene_id)

        # loop through results
        for row in list_drugs:
            # insert the results
            cursor.execute(sql_insert, (row['drug_id'], row['drug_name'], row['gene_id'], row['predicate']))

        # commit every 100
        conn.commit()

        if counter > max_count:
            break

    conn.commit()

