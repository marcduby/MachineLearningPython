
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
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_code = "/Users/mduby/Code/WorkspacePython/"
dir_data = "/home/javaprog/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl
location_servers = dir_code + "MachineLearningPython/DccKP/Translator/Misc/Json/trapiListServices.json"
date_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
location_results = dir_data + "Translator/Workflows/PathwayPpargT2d/Results/" + date_now
location_inputs = dir_code + "MachineLearningPython/DccKP/Translator/Workflows/Json/Queries/Pathways/"
file_result = "{}_{}_results.json"
url_biothings_senmeddb = "https://biothings.ncats.io/semmeddb/query?q=pmid:{}"
max_count = 200

# list of papers
map_papers = {}
map_papers['16150867'] = "3-phosphoinositide-dependent protein kinase-1 activates the peroxisome proliferator-activated receptor-gamma and promotes adipocyte differentiation, Yin "
map_papers['8001151'] = "Stimulation of adipogenesis in fibroblasts by PPAR gamma 2, a lipid-activated transcription factor, Tontonoz"
map_papers['12021175'] = "Gene expression profile of adipocyte differentiation and its regulation by peroxisome proliferator-activated receptor-gamma agonists, Gerhold"
map_papers['10339548'] = "A peroxisome proliferator-activated receptor gamma ligand inhibits adipocyte differentiation. Oberfield"
map_papers['7838715'] = "Adipocyte-specific transcription factor ARF6 is a heterodimeric complex of two nuclear hormone receptors, PPAR gamma and RXR alpha, Tontonoz"
map_papers['10622252'] = "Dominant negative mutations in human PPARgamma associated with severe insulin resistance, diabetes mellitus and hypertension, Barroso"
map_papers['9806549'] = "A Pro12Ala substitution in PPARgamma2 associated with decreased receptor activity, lower body mass index and improved insulin sensitivity, Deeb"
map_papers['25157153'] = "Rare variants in PPARG with decreased activity in adipocyte differentiation are associated with increased risk of type 2 diabetes, Majithia"

def query_biothings(paper_id, log=False):
    ''' 
    find the journal if in the results
    '''
    # initialize
    pubmed_id = 'PMID:' + paper_id
    list_results = []
    is_found = False
    url_query = url_biothings_senmeddb.format(paper_id)

    # log
    if log:
        print("looking for pubmed id: {}".format(url_query))    

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
    map_result = {'id': paper_id, 'predicate': None, 'subject': None, 'subject_type': None, 'object': None, 'object_type': None}
    if json_output:
        if isinstance(json_output, dict):
            if json_output.get('hits'):
                for child in json_output.get('hits'):
                    is_found = True
                    map_result = child.get('predicate')
                    map_result = {'id': paper_id, 'predicate': child.get('predicate'), 'subject': child.get('subject').get('name'), 'subject_type': child.get('subject').get('semantic_type_name'),
                        'object': child.get('object').get('name'), 'object_type': child.get('object').get('semantic_type_name'),}
                    list_results.append(map_result)

    # add to list
    if not is_found:
        list_results.append(map_result)

    # return
    return list_results

if __name__ == "__main__":
    # initialize
    count = 0
    list_result = []

    # loop through the paper ids
    for key, values in map_papers.items():
        # test the max count
        if count < max_count:
            count += 1

            # get the biothings data for the paper
            list_temp = query_biothings(key, log=True)

            # add to the results
            list_result = list_result + list_temp

    # print the results
    print("\n=====results")
    for child in list_result:
        print(child)

    # create dataframe
    df_papers = pd.DataFrame(list_result)
    #temporaly display 999 rows
    with pd.option_context('display.max_rows', 999):
        print (df_papers)


