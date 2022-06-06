
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
dir_code = "/Users/mduby/Code/WorkspacePython/"
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/Users/mduby/Data/Broad/"
dir_data = "/home/javaprog/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl
location_servers = dir_code + "MachineLearningPython/DccKP/Translator/Misc/Json/trapiListServices.json"
date_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
location_results = dir_data + "Translator/Workflows/PathwayPpargT2d/Results/" + date_now
location_inputs = dir_code + "MachineLearningPython/DccKP/Translator/Workflows/Json/Queries/Pathways/"
file_result = "{}_{}_results.json"
dir_kp_result = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/Results/20220511140027"
dir_kp_result = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/Results/20220513103654"
dir_kp_result = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/Results/20220516140102"
dir_kp_result = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/Results/20220516150723"
dir_kp_result = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/Results/20220516204039"

file_reactome = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/GoogleDistancePathways/c2.cp.reactome.v7.5.1.json"
file_go = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/GoogleDistancePathways/c5.go.v7.5.1.json"
file_google_distance = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/GoogleDistancePathways/pathwayGoogleDistanceMin.json"
file_pathways = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/GoogleDistancePathways/pathwayInformation.json"

# initialize
max_count = 50000000000000
count = 0
list_genes = ["BANF1","HMGA1","LIG4","PSIP1","XRCC4","XRCC5","XRCC6"]
list_pathway = []
list_google_distance = []


# methods
def calculate_updated_google_distance(list1, list2, log=False):
    '''
    will calculate the modified google distance between the two lists (intersection divided by min number from both sets)
    '''
    len_intersection = 0
    len_union = 0
    google_distance = 0

    if list1 and list2:
        len_intersection = len(set(list1) & set(list2))
        len_union = min(len(list1), len(list2))
        google_distance = len_intersection/len_union

    # return
    return google_distance

def get_pathway_list(file_pathway, log=False):
    '''
    read a json pathway file and return the list of pathways
    '''
    # initialize
    list_temp = []

    # read in the 
    with open(file_pathway) as file_json: 
        json_reactome = json.load(file_json)

    # get the name, id, gene list
    for key, value in json_reactome.items():
        map_pathway = {'id': value.get('exactSource'), 'name': key, 'list_genes': value.get('geneSymbols')}
        list_temp.append(map_pathway)

    # log
    if log:
        print("for file {} got pathway list of length: {}".format(file_pathway, len(list_temp)))

    # return
    return list_temp
    

if __name__ == "__main__":
    # read in the pathway files
    for file_pathway in [file_go, file_reactome]:
        list_temp = get_pathway_list(file_pathway, log=True)
        list_pathway = list_pathway + list_temp

    # calculate the google distance for all combinations
    for i in range(0, len(list_pathway) - 1):
        for j in range(i, len(list_pathway) - 1):
            # limit for testing
            if count < max_count:
                # get the google distance
                google_distance = calculate_updated_google_distance(list_pathway[i].get('list_genes'), list_pathway[j].get('list_genes'))

                # put the resukt in the result list
                if google_distance > 0.0:
                    count += 1
                    list_google_distance.append({'subject_id': list_pathway[i].get('id'), 'object_id': list_pathway[j].get('id'), 'google_distance_min': google_distance})

                    # print
                    if count % 100000 == 0:
                        print("{} - for {}/{} got GD: {}".format(count, list_pathway[i].get('id'), list_pathway[j].get('id'), google_distance))



    # write out the json result
    map_result = {'results': list_google_distance}
    with open(file_google_distance, 'w') as f:
        json.dump(map_result, f)    
    print("wrote out file: {}".format(file_google_distance))

    # write out the pathway information file
    map_pathways = {'pathways': list_pathway}
    with open(file_pathways, 'w') as f:
        json.dump(map_pathways, f)    
    print("wrote out file: {}".format(file_pathways))


# # initialize
# map_count_edges = {}

# # get the list of files
# print("searching results in directory: {}".format(dir_kp_result))
# for child_file in Path(dir_kp_result).rglob('*.json'):
#     # print("got file: {}".format(child_file))

#     # read the json file result
#     with open(child_file) as file_json: 
#         json_results_file = json.load(file_json)
#         server_name = json_results_file['server_name']
#         query_name = json_results_file['query_name']

#         # find the edge count
#         count_edges = tl.count_trapi_results_edges(json_results_file)
#         if count_edges > 0:
#             map_count_edges[(server_name, query_name)] = count_edges

# # print result
# for server, count in map_count_edges.items():
#     print("Got {} - for {}".format(count, server))



