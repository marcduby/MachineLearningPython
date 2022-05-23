

# imports
import json
import sys 
import logging
import datetime 
import os 
import requests 
import csv 

# constants
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
dir_root = "/Users/mduby"
dir_code = dir_root + "/Code/WorkspacePython/"
dir_data = dir_root + "/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl
location_servers = dir_code + "MachineLearningPython/DccKP/Translator/Misc/Json/trapiListServices.json"
date_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
location_results = dir_data + "Translator/Workflows/MiscQueries/Results/GeneChemicals/" + date_now
location_inputs = dir_data + "Translator/Workflows/MiscQueries/Inputs/GeneChemicals/genes.csv"
location_input_query = dir_code + "MachineLearningPython/DccKP/Translator/Workflows/Json/Queries/MiscGenes/genesChemicals.json"
query_name = "geneChemicals"
file_result = "{}_{}_results.json"
location_result_json = "/Users/mduby/Data/Broad/Translator/Workflows/MiscQueries/Results/GeneChemicals/20220505105254/geneChemicals_biothings-explorer_results.json"


# methods
def get_trapi_result_tuples(json_input, log=True):
    '''
    input trapi result list, returns tuples (subject_id, subject, predicate, object_id, object)
    '''
    # initialize
    list_result = []
    map_nodes = {}

    message = json_input.get('message')
    if message:
        kg = message.get('knowledge_graph')
        if kg:
            # get the nodes
            nodes = kg.get('nodes')
            if nodes:
                for key, values in nodes.items():
                    map_nodes[key] = values.get('name')

            # loop throug the edges
            edges = kg.get('edges')
            if edges:
                for key, value in edges.items():
                    subject = value.get('subject')
                    object = value.get('object')
                    predicate = value.get('predicate')
                    list_result.append((subject, map_nodes.get(subject), predicate, object, map_nodes.get(object)))

    # log
    print("found edges count: {}".format(len(list_result)))

    # return
    return list_result


# load the json results file
with open(location_result_json) as file_json: 
    map_result_json = json.load(file_json)

# find the result tuples
list_result_edges = get_trapi_result_tuples(map_result_json)
print("got list of count: {}\n".format(len(list_result_edges)))
# for row in list_result_edges:
#     print("got result: {}".format(row))


