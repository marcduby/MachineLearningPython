
# imports
import json
import sys 
import logging
import datetime 
import os
import requests 
from pathlib import Path 
import re
import csv

# constants
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/home/javaprog/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl
location_servers = dir_code + "MachineLearningPython/DccKP/Translator/Misc/Json/trapiListServices.json"
date_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
location_results = dir_data + "Translator/Workflows/PathwayPpargT2d/Results/" + date_now
location_inputs = dir_code + "MachineLearningPython/DccKP/Translator/Workflows/Json/Queries/Pathways/"
file_result = "{}_{}_results.json"
dir_kp_result = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/Results/20220511140027"
dir_kp_result = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/Results/20220513103246"
# location_ara_results = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/Results/20220511134655/"


# initialize
map_count_edges = {}

# get the list of files
print("searching results in directory: {}".format(dir_kp_result))
for child_file in Path(dir_kp_result).rglob('*.json'):
    # print("got file: {}".format(child_file))

    # read the json file result
    with open(child_file) as file_json: 
        json_results_file = json.load(file_json)
        server_name = json_results_file['server_name']
        query_name = json_results_file['query_name']

        # find the edge count
        count_edges = tl.count_trapi_results_edges(json_results_file)
        if count_edges > 0:
            map_count_edges[(server_name, query_name)] = count_edges

# print result
for server, count in map_count_edges.items():
    print("Got {} - for {}".format(count, server))



