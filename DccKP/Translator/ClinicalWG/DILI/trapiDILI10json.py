# -*- coding: utf-8 -*-

# imports
import requests
import json
import pandas as pd
import numpy as np

# methods
def trapi_query(json_input, url):
    ''' copied from Casey Ta's work '''
    # return requests.post('https://cohd.io/api/translator/query', 
    #                      json=json.loads(json_str))
    # print(json_input)

    # genetics KP url
    return requests.post(url, json=json_input)

def get_curie_synonyms(curie_input, prefix_list=None):
  ''' will call the curie normalizer and return the curie name and a list of only the matching prefixes from the prefix list provided '''
  url_normalizer = "https://nodenormalization-sri.renci.org/get_normalized_nodes?curie={}"
  list_result = []
  curie_name = None

  # call the service
  url_call = url_normalizer.format(curie_input)
  response = requests.get(url_call)
  json_response = response.json()

  # get the list of curies
  if json_response.get(curie_input):
    curie_name = json_response.get(curie_input).get('id').get('label')
    for item in json_response[curie_input]['equivalent_identifiers']:
      list_result.append(item['identifier'])

    # if a prefix list provided, filter with it
    if prefix_list:
      list_new = []
      for item in list_result:
        if item.split(':')[0] in prefix_list:
          list_new.append(item)
      list_result = list_new

  # return
  return curie_name, list_result

def build_result_list(result, original_curie, curie_name):
  ''' will build a list of results '''
  # initialize
  result_list = []
  # print(result['message']['knowledge_graph'])

  # get the root of the result
  edges = result['message']['knowledge_graph']['edges']
  nodes = result['message']['knowledge_graph']['nodes']

  # get the edge data
  for key in edges.keys():
    # print("key {}".format(key))
    edge = edges.get(key)
    result = {'curie': edge.get('object'), 'predicate': edge.get('predicate')}
    result_list.append(result)

  # get the node data
  for item in result_list:
    node = nodes[item['curie']]
    item['name'] = node.get('name')
    item['category'] = node.get('category')
    item['original_curie'] = original_curie
    item['original_curie_name'] = curie_name

  # return
  return result_list

def build_result_list(result, original_curie, curie_name):
  ''' will build a list of results '''
  # initialize
  result_list = []
  # print(result['message']['knowledge_graph'])

  # get the root of the result
  edges = result['message']['knowledge_graph']['edges']
  nodes = result['message']['knowledge_graph']['nodes']

  # get the edge data
  for key in edges.keys():
    # print("key {}".format(key))
    edge = edges.get(key)
    result = {'object_curie': edge.get('object'), 'object_predicate': edge.get('predicate')}

    attributes = edge.get('attributes')
    if attributes:
      # list_attributes = [(att.get('original_attribute_name'), att.get('value'), att.get('attribute_source')) if att.get('original_attribute_name') in ['article', 'actions']]
      # list_attributes = [(att.get('original_attribute_name'), att.get('value'), att.get('attribute_source')) for att in attributes if att.get('original_attribute_name') in ['actions', 'article']]
      list_attributes = [(att.get('original_attribute_name'), att.get('value'), att.get('attribute_source')) for att in attributes]
      result['object_attributes'] = list_attributes

    result_list.append(result)

  # get the node data
  for item in result_list:
    node = nodes[item['object_curie']]
    item['object_name'] = node.get('name')
    item['object_category'] = node.get('category')
    item['original_curie'] = original_curie
    item['original_curie_name'] = curie_name

  # return
  return result_list


# constants
url_molepro = 'https://translator.broadinstitute.org/molepro/trapi/v1.0/query'
url_genetics = 'https://translator.broadinstitute.org/genetics_provider/trapi/v1.0/query'
url_genetics = 'http://localhost:7001/genetics_provider/trapi/v1.0/query'

# files
file_clinical = '/home/javaprog/Code/PythonWorkspace/MachineLearningPython/DccKP/Translator/ClinicalWG/DILI/exposuresResult.json'
file_genetis_output = '/home/javaprog/Code/PythonWorkspace/MachineLearningPython/DccKP/Translator/ClinicalWG/DILI/step2b_Genetics_Output.txt'

# add in extra genetics KP lab values that we have per Jennifer H's suggestion
# genetics KP liver related phen otypes/diseases
# read the file
# read in the file with the input curies from the previous step
# with open(f'{file_clinical}', 'r') as file_dili:
#     json_data = json.load(file_dili)
#     for 
#     list_inputs = file_dili.read().split('\n')
#     print("list size {}".format(len(list_inputs)))

# read the file
map_exposures_synonyms = {}
with open(f'{file_clinical}', 'r') as file_exposures_dili:
    json_data = json.load(file_exposures_dili)
    list_inputs = json_data.get('message').get('knowledge_graph').get('nodes')
    print("got node count of {}".format(len(list_inputs)))
    # for each disease, record the list of exposures kb provided synonyms
    for item in list_inputs:
      list_attributes = json_data.get('message').get('knowledge_graph').get('nodes').get(item).get('attributes')
      name = json_data.get('message').get('knowledge_graph').get('nodes').get(item).get('name')
      for att in list_attributes:
        if att.get('name') == 'equivalent_identifiers':
          map_exposures_synonyms[item] = {'name': name, 'synonyms': att.get('value')}
print("got identifier map length of {}".format(len(map_exposures_synonyms.keys())))
phenotype = 'MESH:D019934'
print("for {} got synonym list of size {}".format(phenotype, len(map_exposures_synonyms.get(phenotype).get('synonyms'))))

# list_inputs = ['EFO:0000289'] + list_inputs
# print(list_inputs)
# list_inputs = ['MONDO:0018106']
# list_inputs = ['EFO:0004611']
# add to the input list
# print("list size {} and data: \n{}".format(len(list_noonan), list_noonan))

# build the json


# build the json
# TODO - figure out parametrized code for this
input_json =   {
    "message": {
        "query_graph": {
            "edges": {
                "e00": {
                    "subject": "n00",
                    "object": "n01",
                    "predicate": "biolink:condition_associated_with_gene"
                }
            },
            "nodes": {
                "n00": {
                    "id": "MONDO:0019600",
                    "category": "biolink:Disease"
                },
                "n01": {
                    "category": "biolink:Gene"
                }
            }
        }
    }
}


# 1 - simple gene to drugs that treat it
# build a map of call results
genetics_results = {}
list_category = ['biolink:Disease', 'biolink:PhenotypicFeature']
list_gene = []
pvalue_cutoff = 0.0000000025

# build a map of call results using the exposures KP provided synonyms 'map_exposures_synonyms'
exposures_genetics_results = {}
for input_curie in map_exposures_synonyms.keys():
  # initialize 
  result_list = []
  # get the curie_name and synonym list for each input
  prefix_list = ['EFO', 'MONDO']
  curie_name = map_exposures_synonyms.get(input_curie).get('name')
  list_synonym = [syn for syn in map_exposures_synonyms.get(input_curie).get('synonyms') if syn.split(':')[0] in prefix_list]
  # curie_name, list_synonym = get_curie_synonyms(input_curie, ['EFO', 'MONDO'])
  print(f"\n== for curie {input_curie} with name {curie_name} got (EFO/MONDO) synonym list of size {len(list_synonym)}")
  for curie in list_synonym:
    # set the curie
    input_json['message']['query_graph']['nodes']['n00']['id'] = curie
    # call the rest service
    response = trapi_query(input_json, url_genetics)
    # get the list for the response
    result_list = build_result_list(response.json(), input_curie, curie_name)
    # log
    # print("genetics query for {}".format(input_json['message']['query_graph']['nodes']['n00']['id']))
    print("genetics query synonym curie {} got trapi results of size {}".format(curie, len(result_list)))
    for item in result_list:
      print(f'\t {item.get("object_predicate")} - {item.get("object_curie")} - {item.get("object_name")} - {item.get("object_attributes")}')
      list_gene.append(item.get("object_curie"))
    # print
    # for item in result_list:
    #   print(item)
  # add results to map
  exposures_genetics_results[input_curie] = result_list

# for input_curie in list_inputs:
#   for category in list_category:
#     # initialize 
#     result_list = []

#     # get the curie_name and synonym list for each input
#     curie_name, list_synonym = get_curie_synonyms(input_curie, ['EFO', 'MONDO'])

#     # set the curie
#     input_json['message']['query_graph']['nodes']['n00']['id'] = input_curie
#     input_json['message']['query_graph']['nodes']['n00']['category'] = category

#     # log query 
#     # print(input_json)

#     # call the rest service
#     response = trapi_query(input_json, url_genetics)
#     # print("***********************")
#     # print(response.json())

#     # get the list for the response
#     temp_result_list = build_result_list(response.json(), input_curie, curie_name)

#     # print
#     print(f'for {input_curie} - {curie_name}, using category {category}')
#     for item in temp_result_list:
#       print(f'\t {item.get("object_predicate")} - {item.get("object_curie")} - {item.get("object_name")} - {item.get("object_attributes")}')
#       list_gene.append(item.get("object_curie"))

#       # if item.get('object_attributes'):
#       #   for att in item.get('object_attributes'):
#       #     print(f'\t\t{att}')
#     # log
#     # print("genetics query for {}".format(input_json['message']['query_graph']['nodes']['n00']['id']))
#     # print("molepro query synonym curie {} got trapi results of size {}\n".format(curie, len(result_list)))
#     # print(f"{curie}\n\n{temp_result_list[0]}\n\n")

#     # print
#     # for item in result_list:
#     #   print(item)
#     result_list += temp_result_list

  # # add results to map
  # genetics_results[input_curie] = result_list

# make sure list is unique genes
list_gene = list(set(list_gene))
print(f'gene list of size m{len(list_gene)} and data\n: {list_gene}')

# write out the results
with open(f'{file_genetis_output}', 'w') as file_dili:
  for item in list_gene:
    file_dili.write(f'{item}\n')
# f = open(file_genetis_output, "a")
# f.write("Now the file has more content!")
# f.close()




















