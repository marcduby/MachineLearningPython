# -*- coding: utf-8 -*-

# imports
import requests
import json
from IPython.display import HTML
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


# constants
url_molepro = 'https://translator.broadinstitute.org/molepro/trapi/v1.0/query'
url_genetics = 'https://translator.broadinstitute.org/genetics_provider/trapi/v1.0/query'


# add in extra genetics KP lab values that we have per Jennifer H's suggestion
# genetics KP liver related phen otypes/diseases
list_noonan = ['MONDO:0008104']

# add to the input list
print("list size {} and data: \n{}".format(len(list_noonan), list_noonan))


# build the json
# TODO - figure out parametrized code for this
input_json = {
    "message": {
        "query_graph": {
            "edges": {
                "e00": {
                    "subject": "n00",
                    "object": "n01"
                }
            },
            "nodes": {
                "n00": {
                    "id": "EFO:0000289"
                },
                "n01": {
                }
            }
        }
    }
}


# # build a map of call results
# genetics_results = {}
# for input_curie in list_noonan:
#   # initialize 
#   result_list = []

#   # get the curie_name and synonym list for each input
#   curie_name, list_synonym = get_curie_synonyms(input_curie, ['EFO', 'MONDO'])
#   print(f"\n== for curie {input_curie} with name {curie_name} got (EFO/MONDO) synonym list of size {len(list_synonym)}")
#   for curie in list_synonym:
#     # set the curie
#     input_json['message']['query_graph']['nodes']['n00']['id'] = curie

#     # call the rest service
#     response = trapi_query(input_json, url_genetics)
#     # print("***********************")
#     # print(response.json())

#     # get the list for the response
#     result_list = build_result_list(response.json(), input_curie, curie_name)

#     # log
#     # print("genetics query for {}".format(input_json['message']['query_graph']['nodes']['n00']['id']))
#     print("genetics query synonym curie {} got trapi results of size {}".format(curie, len(result_list)))

#     # print
#     # for item in result_list:
#     #   print(item)

#   # add results to map
#   genetics_results[input_curie] = result_list


# build the json
# TODO - figure out parametrized code for this
input_json = {
    "message": {
        "query_graph": {
            "edges": {
                "e00": {
                    "subject": "n00",
                    "object": "n01",
                    "predicate": "biolink:treated_by"
                }
            },
            "nodes": {
                "n00": {
                    "id": "NCBIGene:1803",
                    "category": "biolink:Disease"
                },
                "n01": {
                    "category": "biolink:ChemicalSubstance"
                }
            }
        }
    }
}

# look for drugs for gene
list_predicates_gene_chem = ['biolink:affected_by', 'biolink:correlated_with']
list_gene = ['HGNC:11364']
list_predicates_disease = ['biolink:treated_by']

# 1 - simple gene to drugs that treat it
# build a map of call results
molepro_results = {}

for input_curie in list_gene:
  # initialize 
  result_list = []

  # get the curie_name and synonym list for each input
  curie_name, list_synonym = get_curie_synonyms(input_curie)
  # print(f"\n== for curie {input_curie} with name {curie_name} got (EFO/MONDO) synonym list of size {len(list_synonym)}")
  for curie in list_synonym:
    for pred in list_predicates_gene_chem:
      # set the curie
      input_json['message']['query_graph']['nodes']['n00']['id'] = curie
      input_json['message']['query_graph']['nodes']['n00']['category'] = 'biolink:Gene'
      input_json['message']['query_graph']['nodes']['n01']['category'] = 'biolink:ChemicalSubstance'
      input_json['message']['query_graph']['edges']['e00']['predicate'] = pred

      # log query 
      # print(input_json)

      # call the rest service
      response = trapi_query(input_json, url_molepro)
      print("***********************")
      print(response.json())

      # get the list for the response
      temp_result_list = build_result_list(response.json(), input_curie, curie_name)

      # print
      print(f'for {input_curie} - {curie_name}, using synonym curie {curie}')
      for item in temp_result_list:
        print(f'\t {item.get("predicate")} - {item.get("curie")} - {item.get("name")}')
      # log
      # print("genetics query for {}".format(input_json['message']['query_graph']['nodes']['n00']['id']))
      # print("molepro query synonym curie {} got trapi results of size {}\n".format(curie, len(result_list)))
      print(f"{curie}\n\n{temp_result_list[0]}\n\n")

      # print
      # for item in result_list:
      #   print(item)
      result_list += temp_result_list

  # add results to map
  molepro_results[input_curie] = result_list




















"""**Test code - scratch**"""

# test the get_curie_synonyms method
curie_name, list_result = get_curie_synonyms('EFO:0000289', ['EFO', 'MONDO'])
for item in list_result:
  print("for {} got curie {}".format(curie_name, item))