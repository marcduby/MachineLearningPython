
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

# initialize
count = 0
count_max = 500

# functions
def read_translator_csv(file_csv, log=False):
    '''
    read the csv file, return a list of ontology ids
    '''
    # initialize
    list_ids = []
    count = 0

    # read the file
    with open(file_csv, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if count != 0:
                list_ids.append(row)
            count += 1

    # log
    print("got number of ids: {}".format(len(list_ids)))

    # return
    return list_ids

def build_input(json_input, list_subjects, log=False):
    '''
    returns a modified query with new ids
    '''
    # set the gene list on the query
    message = json_input.get('message')
    nodes = message.get('query_graph').get('nodes')
    subject = nodes.get('gene')
    subject['ids'] = list_genes

    # log
    print("using input: {}".format(json_input))

    # return
    return json_input

# main
if __name__ == "__main__":
    # read the file
    with open(location_servers) as file_json: 
        json_servers = json.load(file_json)

    # load the trapi kps
    list_servers = tl.get_trapi_kps(json_servers)
    print("got {} KPs".format(len(list_servers)))
    print("datetime: {}".format(date_now))

    # create the results directory
    os.mkdir(location_results)

    # read the inputs
    list_gene_tuples = read_translator_csv(location_inputs)
    print("got input gene from file: {}".format(len(list_gene_tuples)))

    # build map of cluster and genes
    map_cluter_gene_list = {}
    print(list_gene_tuples)
    for [cluster, gene] in list_gene_tuples:
        if not map_cluter_gene_list.get(cluster):
            map_cluter_gene_list[cluster] = []
        map_cluter_gene_list[cluster].append(gene)
    print("got input gene clusters: {}".format(map_cluter_gene_list))
    # list_genes = ['NCBIGene:27349']

    for cluster, list_input_genes in map_cluter_gene_list.items():
        # translate to ontology ids
        list_gene_translated_tuples = tl.translate_to_ontology_id(list_input_genes, 'NCBIGene', sort_by_ontology=True)
        list_genes = [row_gene for (row_name, row_gene) in list_gene_translated_tuples]
        print("\ngot number input translated gene of: {}".format(len(list_genes)))

        # load the json inputs
        map_input_json = {}
        with open(location_input_query) as file_json: 
            map_input_json[query_name] = json.load(file_json)
        json_input = build_input(map_input_json[query_name], list_genes)

        # loop through servers
        for trapi in list_servers:
            # increment count
            count += 1

            # loop through input queries
            if count < count_max:
                # make the result directory
                server_name = trapi['info'].split(":")[1]

                # loop through input queries
                for name_input, json_input in map_input_json.items():
                    url_query = trapi['url'] + "/query"
                    print("\n{} = got input name: {} to {}".format(count, name_input, url_query))
                    response = requests.post(url_query, json=json_input)

                    try:
                        json_output = response.json()
                        print("got result: \n{}".format(json_output))
                    except ValueError:
                        print("GOT ERROR: skipping")
                        continue

                    # add the type of query and service name to the json
                    json_output['server_name'] = server_name
                    json_output['query_name'] = name_input

                    # write out file
                    file_output = location_results + "/" + file_result.format(name_input, server_name)
                    with open(file_output, 'w') as f:
                        print("writing out to file: {}".format(file_output))
                        json.dump(json_output, f, ensure_ascii=False, indent=2)




