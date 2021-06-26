
# GOAL
# for each rare disease
# - find the one associated gene
# - find other common diseases they are related to 
#
# the rare disease will serve as a basis for research into the common disease


# imports
import requests
import json 
import time
import pandas as pd 

# constants
url_spoke = "https://spokekp.healthdatascience.cloud/api/v1.1/query"
file_rare_disease = '/home/javaprog/Data/Broad/Translator/RareDisease/Gene_DCC_GARD_RareDiseases.csv'
flag_overwrite = False 

# methods
def build_trapi(subject_id, subject_type, object_type, predicate, log=False):
    ''' build the trapi paylaod map '''
    # initialize
    map_trapi = {}

    # build the map
    subject = {"ids": [subject_id], "categories": [subject_type]}
    object = {"categories": [object_type]}
    nodes = {"n00": subject, "n01": object}
    edges = {"e00": {"subject": "n00", "object": "n01", "predicates": [predicate]}}
    query_graph = {"edges": edges, "nodes": nodes}
    map_trapi = {"message": {"query_graph": query_graph}}

    # log
    if log:
        print("got query: \n{}".format(map_trapi))

    # return
    return map_trapi

def parse_result(json, log=False):
    ''' parse the results from the json response '''
    # intialize
    array_genes = []

    # get the result array
    if json.get('message').get('knowledge_graph'):
        if json.get('message').get('knowledge_graph').get('nodes'):
            print(json.get('message').get('knowledge_graph').get('nodes'))
            for key, value in json.get('message').get('knowledge_graph').get('nodes').items():
                if "NCBIGene" in key:
                    array_genes.append((key, value.get('name')))

    # log
    if log:
        print("found gene list of size {}: {}".format(len(array_genes), array_genes))

    # return
    return array_genes

def call_trapi(url_trapi, subject_id, subject_type, object_type, predicate, log=False):
    ''' method to build the json payload and call the trapi service '''
    # test the trapi building
    json_payload = build_trapi(subject_id=subject_id, subject_type=subject_type, object_type=object_type, predicate=predicate, log=log)

    # call the trapi service
    response = requests.post(url_spoke, json=json_payload)
    json_result = response.json()

    # print the result
    array_genes = parse_result(json=json_result, log=log)

    # return
    return array_genes

if __name__ == "__main__":
    # print the result
    array_genes = call_trapi(url_trapi=url_spoke, subject_id="MONDO:0008757", subject_type="biolink:Disease", object_type="biolink:Gene", predicate="biolink:condition_associated_with_gene", log=True)

    # read in the pandas data
    df_rare_disease = pd.read_csv(file_rare_disease, sep=',', header=0)
    print("after reading: \n{}".format(df_rare_disease.info()))

    # loop through rows and look for gene associated to ontology id 
    count = 0
    for index, row in df_rare_disease.iterrows():
        ontology = row['ontology']
        if not pd.isnull(ontology) and pd.isnull(row['gene_check']):
            if flag_overwrite or pd.isnull(row['gene_id']):
                # log
                print("look for gene for ontology id: {}".format(ontology))
                count += 1

                # # find ontology
                # json_payload = build_trapi(subject_id="MONDO:0008757", subject_type="biolink:Disease", object_type="biolink:Gene", predicate="biolink:condition_associated_with_gene", log=True)

                # # call the trapi service
                # response = requests.post(url_spoke, json=json_payload)
                # json_result = response.json()

                # print the result
                array_genes = call_trapi(url_trapi=url_spoke, subject_id=ontology, subject_type="biolink:Disease", object_type="biolink:Gene", predicate="biolink:condition_associated_with_gene", log=True)

                # if found, log and set
                if array_genes is not None and len(array_genes) == 1:
                    gene_id, gene_name = array_genes[0]
                    print("found gene {} for ontology: {}\n".format(gene_id, ontology))
                    df_rare_disease.loc[df_rare_disease['ontology'] == ontology, ['gene_id']] = gene_id
                    df_rare_disease.loc[df_rare_disease['ontology'] == ontology, ['gene_name']] = gene_name

                # tag that checked
                df_rare_disease.loc[df_rare_disease['ontology'] == ontology, ['gene_check']] = "yes"

                # break if count reached
                if count%10 == 0:
                    print("{} - data saved to file".format(count))
                    df_rare_disease.to_csv(file_rare_disease, sep=',', index=False)
                    # break

                # sleep for throttling avoidance
                time.sleep(7)
        
    # log
    print("\nafter updating: \n{}".format(df_rare_disease.info()))

    # write out results 
    # df_rare_disease.to_csv(file_test_rare_disease, sep=',')
    df_rare_disease.to_csv(file_rare_disease, sep=',', index=False)

