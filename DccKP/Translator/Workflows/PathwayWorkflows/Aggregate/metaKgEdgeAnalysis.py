
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
import pandas as pd

# constants
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
dir_code = "/Users/mduby/Code/WorkspacePython/"
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/Users/mduby//Data/Broad/"
dir_data = "/home/javaprog/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl
location_servers = dir_code + "MachineLearningPython/DccKP/Translator/Misc/Json/trapiListServices.json"
date_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
location_results = dir_data + "Translator/Workflows/PathwayPpargT2d/SenmedDb/"
file_result = location_results + "metaKgEdges.csv"
url_metakg = "https://smart-api.info/api/metakg?subject={}&object={}"
max_count = 200



def query_metakg(subject, object, log=False):
    ''' 
    query the metakg for the edges specified
    '''
    # initialize
    map_results = {}
    list_results = []
    is_found = False
    url_query = url_metakg.format(subject, object)

    # log
    if log:
        print("looking for query: {}".format(url_query))    

    # query the service
    response = requests.get(url_query)

    # try and catch exception
    try:
        json_output = response.json()
        # if log:
        #     print("got result: \n{}".format(json_output))
    except ValueError:
        print("GOT ERROR: skipping")

    # pick put the data
    if json_output.get('associations'):
        for row in json_output.get('associations'):
            row_subject = row.get('subject')
            row_object = row.get('object')
            row_predicate = row.get('predicate')

            # log
            if log:
                print("got triple: {}, {}, {}".format(row_subject, row_object, row_predicate))

            # if row.get('api').get('x-translator').get('component') == 'KP':

            #     # create key if not in map
            #     if not map_results.get("{} - {} - {}".format(row_subject, row_predicate, row_object)):
            #         # map_results[row_predicate] = []
            #         map_results["{} - {} - {}".format(row_subject, row_predicate, row_object)] = []

            #     # add server name to list for that tuple
            #     # map_results[row_predicate].append(row.get('api').get('name'))
            #     map_results["{} - {} - {}".format(row_subject, row_predicate, row_object)].append(row.get('api').get('name'))

            if row.get('api').get('x-translator').get('component') == 'KP':

                # create key if not in map
                if not map_results.get(row_predicate):
                    # map_results[row_predicate] = []
                    map_results[row_predicate] = []

                # add server name to list for that tuple
                # map_results[row_predicate].append(row.get('api').get('name'))
                map_results[row_predicate].append(row.get('api').get('name'))

    # make list unique
    for key, values in map_results.items():
        list_unique = list(set(values))
        map_results[key] = list_unique

    # return
    return map_results

if __name__ == "__main__":
    # initialize
    count = 0
    map_results = []
    list_count = []
    list_servers = []
    subject = 'Pathway'
    object = 'Gene'

    # get the data
    map_results = query_metakg(subject, object, log=True)

    # create dataframe
    for key, values in map_results.items():
        list_count.append({'predicate': key, 'count': len(values)})
        list_servers = list_servers + values

    df_edges = pd.DataFrame(list_count)
    #temporaly display 999 rows
    with pd.option_context('display.max_rows', 999):
        print (df_edges.to_string(index=False))

    list_servers = list(set(list_servers))
    print("got {} servers".format(len(list_servers)))

    # write out the file
    df_edges.to_csv(file_result, sep='\t')
    print("wrote out the file to: {}".format(file_result))

