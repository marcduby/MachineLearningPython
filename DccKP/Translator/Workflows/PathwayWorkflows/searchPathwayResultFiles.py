
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
dir_result = dir_data + "Translator/Workflows/PathwayPpargT2d/Results/20220427224745"
location_ara_results = dir_data + "Translator/Workflows/PathwayPpargT2d/AraResults/20220429105914/"

# list of papers
map_papers = {}
map_papers['8001151'] = "Stimulation of adipogenesis in fibroblasts by PPAR gamma 2, a lipid-activated transcription factor, Tontonoz"
map_papers['16150867'] = "3-phosphoinositide-dependent protein kinase-1 activates the peroxisome proliferator-activated receptor-gamma and promotes adipocyte differentiation, Yin "
map_papers['12021175'] = "Gene expression profile of adipocyte differentiation and its regulation by peroxisome proliferator-activated receptor-gamma agonists, Gerhold"
map_papers['10339548'] = "A peroxisome proliferator-activated receptor gamma ligand inhibits adipocyte differentiation. Oberfield"
map_papers['7838715'] = "Adipocyte-specific transcription factor ARF6 is a heterodimeric complex of two nuclear hormone receptors, PPAR gamma and RXR alpha, Tontonoz"
map_papers['10622252'] = "Dominant negative mutations in human PPARgamma associated with severe insulin resistance, diabetes mellitus and hypertension, Barroso"
map_papers['9806549'] = "A Pro12Ala substitution in PPARgamma2 associated with decreased receptor activity, lower body mass index and improved insulin sensitivity, Deeb"
map_papers['25157153'] = "Rare variants in PPARG with decreased activity in adipocyte differentiation are associated with increased risk of type 2 diabetes, Majithia"

def find_publication(paper_id, element, log=False):
    ''' 
    find the journal if in the results
    '''
    # initialize
    is_contained = False
    pubmed_id = 'PMID:' + paper_id

    # log
    if log:
        print("looking for pubmed id: {}".format(pubmed_id))    

    # loop of dict or list
    if isinstance(element, list):
        for child in element:
            if find_publication(paper_id, child):
                return True
    elif isinstance(element, dict):
        for child in element.values():
            if find_publication(paper_id, child):
                return True
    else:
        is_contained = pubmed_id in str(element)

    # return
    return is_contained


# initialize
map_journal_references = {}
for key in map_papers.keys():
    map_journal_references[key] = []

# get the list of files
print("searching results in directory: {}".format(dir_result))
for child_file in Path(dir_result).rglob('*.json'):
    # print("got file: {}".format(child_file))

    # read the json file result
    with open(child_file) as file_json: 
        json_results_file = json.load(file_json)
        server_name = json_results_file['server_name']
        query_name = json_results_file['query_name']

        # loop through pubmed ids and search 
        for paper_id, title in map_papers.items():
            if find_publication(paper_id, json_results_file):
                # if found, add to map and break
                map_journal_references[paper_id].append({'server': server_name, 'query': query_name})

# print result
for paper_id, reference in map_journal_references.items():
    print("{} - {}".format(map_papers[paper_id], reference))

# find ara pubmed ids
# read in the ARA results
file_ara_results = location_ara_results + "twohop_arax_results.json"
with open(file_ara_results) as file_json: 
    json_results = json.load(file_json)
list_papers = tl.find_all_instances_string("PMID", json_results)
list_papers = list(set(list_papers))
print("\nfound PUBMED IDs: {}".format(list_papers))
print("found {} paper links".format(len(list_papers)))

# remove the PMID prefix, sort and write to file
# list_papers = {re.search('PMID:(\d*)', item).group(0) for item in list_papers}
list_ids = []
for item in list_papers:
    string_search = re.search('PMID:(\d*)', item)
    if string_search:
        string_id = string_search.group(0)
        list_ids.append(string_id)

# list_papers.sort()
list_ids = [item.replace('PMID:', '') for item in list_ids]
list_ids = list(set(list_ids))
list_ids = [item for item in list_ids if item]
list_ids.sort()
print("\nfound PUBMED IDs: {}".format(list_ids))
print("found {} paper links".format(len(list_ids)))

for item in list_ids:
    print("item: {}".format(item))

location_csv_results = location_ara_results + "arax_pubmed.csv"
with open(location_csv_results, 'w', newline='') as file_csv_results:
    wr = csv.writer(file_csv_results, quoting=csv.QUOTE_ALL, delimiter='\n')
    wr.writerow(list_ids)
    print("wrote csv file to {}".format(location_csv_results))



# with open(..., 'w', newline='') as myfile:
#      wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#      wr.writerow(mylist)


