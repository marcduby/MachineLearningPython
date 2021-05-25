# -*- coding: utf-8 -*-

# imports
import requests
import json
import pandas as pd
import numpy as np
import argparse 

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
    result = {'object_curie': edge.get('object'), 'subject_curie': edge.get('subject'), 'object_predicate': edge.get('predicate')}

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

def process_inputs(list_inputs):
  # build a map of call results
  genetics_results = {}
  gene_result_list = []

  for input_curie in list_inputs:
    # initialize 
    result_list = []

    # get the curie_name and synonym list for each input
    curie_name, list_synonym = get_curie_synonyms(input_curie, ['EFO', 'MONDO'])

    # print(f"\n== for curie {input_curie} with name {curie_name} got (EFO/MONDO) synonym list of size {len(list_synonym)}")
    for curie in list_synonym:
      # set the curie
      input_json['message']['query_graph']['nodes']['n00']['ids'] = [curie]

      # log query 
      # print(input_json)

      # call the rest service
      response = trapi_query(input_json, url_genetics)
      # print("***********************")
      # print(response.json())

      # get the list for the response
      temp_result_list = build_result_list(response.json(), input_curie, curie_name)

      # print
      print(f'for {input_curie} - {curie_name}, using synonym curie {curie}')
      for item in temp_result_list:
        print(f'\t {item.get("subject_curie")} - {item.get("object_predicate")} - {item.get("object_curie")} - {item.get("object_name")}')
        gene_result_list.append(item.get('object_curie'))
      # log
      # print("genetics query for {}".format(input_json['message']['query_graph']['nodes']['n00']['id']))
      # print("molepro query synonym curie {} got trapi results of size {}\n".format(curie, len(result_list)))
      # print(f"{curie}\n\n{temp_result_list[0]}\n\n")

      # print
      # for item in result_list:
      #   print(item)
      result_list += temp_result_list

    # add results to map
    genetics_results[input_curie] = result_list

  # make sure list is unique
  gene_result_list = list(set(gene_result_list))
  print("result gene list is {}".format(gene_result_list))

  # return
  return gene_result_list



if __name__ == "__main__":
  # configure argparser
  parser = argparse.ArgumentParser("script to clone the dev bioindex data to the prod machine")
  # add the arguments
  parser.add_argument('-i', '--input', help='the input file', required=True)
  parser.add_argument('-o', '--output', help='the output file', required=True)
  parser.add_argument('-e', '--expanded', help='the expanded output file', required=False)
  # get the args
  args = vars(parser.parse_args())


  # constants
  url_molepro = 'https://translator.broadinstitute.org/molepro/trapi/v1.1/query'
  url_genetics = 'https://translator.broadinstitute.org/genetics_provider/trapi/v1.1/query'

  # files
  file_clinical = '/home/javaprog/Code/PythonWorkspace/MachineLearningPython/DccKP/Translator/ClinicalWG/DILI/step1a.txt'
  file_genetics_output = '/home/javaprog/Code/PythonWorkspace/MachineLearningPython/DccKP/Translator/ClinicalWG/DILI/step2a_Genetics_Output.txt'
  file_genetics_expanded_output = None
  if args['input'] is not None:
    file_clinical = args['input']
  if args['output'] is not None:
    file_genetics_output = args['output']
  if args['expanded'] is not None:
    file_genetics_expanded_output = args['expanded']


  # add in extra genetics KP lab values that we have per Jennifer H's suggestion
  # genetics KP liver related phen otypes/diseases
  # read the file
  map_exposures_synonyms = {}
  expanded_list_inputs = []
  with open(f'{file_clinical}', 'r') as file_exposures_dili:
      json_data = json.load(file_exposures_dili)
      list_inputs = json_data.get('message').get('knowledge_graph').get('nodes')
      print("got node count of {}".format(len(list_inputs)))
      # for each disease, record the list of exposures kb provided synonyms
      for item in list_inputs:
        list_attributes = json_data.get('message').get('knowledge_graph').get('nodes').get(item).get('attributes')
        name = json_data.get('message').get('knowledge_graph').get('nodes').get(item).get('name')
        if list_attributes:
          for att in list_attributes:
            if att.get('name') == 'equivalent_identifiers':
              expanded_list_inputs += att.get('value')
              # map_exposures_synonyms[item] = {'name': name, 'synonyms': att.get('value')}
  # print("got identifier map length of {}".format(len(map_exposures_synonyms.keys())))
  # phenotype = 'MESH:D019934'
  # print("for {} got synonym list of size {}".format(phenotype, len(map_exposures_synonyms.get(phenotype).get('synonyms'))))

  # build the json
  # TODO - figure out parametrized code for this
  input_json =   {
      "message": {
        "query_graph": {
          "edges": {
            "e00": {
              "subject": "n00",
              "object": "n01",
              "predicates": ["biolink:condition_associated_with_gene"]
            }
          },
          "nodes": {
            "n00": {
              "ids": ["MONDO:0018106"],
              "categories": ["biolink:Disease", "biolink:PhenotypicFeature"]
            },
            "n01": {
              "categories": ["biolink:Gene"]
            }
          }
        }
      }
    }


  # # build a map of call results
  # genetics_results = {}
  # gene_result_list = []

  # for input_curie in list_inputs:
  #   # initialize 
  #   result_list = []

  #   # get the curie_name and synonym list for each input
  #   curie_name, list_synonym = get_curie_synonyms(input_curie, ['EFO', 'MONDO'])

  #   # skip the synonyms for now 
  #   # list_synonym = [input_curie]

  #   # print(f"\n== for curie {input_curie} with name {curie_name} got (EFO/MONDO) synonym list of size {len(list_synonym)}")
  #   for curie in list_synonym:
  #     # set the curie
  #     input_json['message']['query_graph']['nodes']['n00']['ids'] = [curie]

  #     # log query 
  #     # print(input_json)

  #     # call the rest service
  #     response = trapi_query(input_json, url_genetics)
  #     # print("***********************")
  #     # print(response.json())

  #     # get the list for the response
  #     temp_result_list = build_result_list(response.json(), input_curie, curie_name)

  #     # print
  #     print(f'for {input_curie} - {curie_name}, using synonym curie {curie}')
  #     for item in temp_result_list:
  #       print(f'\t {item.get("subject_curie")} - {item.get("object_predicate")} - {item.get("object_curie")} - {item.get("object_name")}')
  #       gene_result_list.append(item.get('object_curie'))
  #     # log
  #     # print("genetics query for {}".format(input_json['message']['query_graph']['nodes']['n00']['id']))
  #     # print("molepro query synonym curie {} got trapi results of size {}\n".format(curie, len(result_list)))
  #     # print(f"{curie}\n\n{temp_result_list[0]}\n\n")

  #     # print
  #     # for item in result_list:
  #     #   print(item)
  #     result_list += temp_result_list

  #   # add results to map
  #   genetics_results[input_curie] = result_list

  # # make sure list is unique
  # gene_result_list = list(set(gene_result_list))
  # print("result gene list is {}".format(gene_result_list))

  # generate a file for the regular inputs
  gene_result_list = process_inputs(list_inputs)

  # write out the results
  with open(file_genetics_output, "w") as out_file:
    for item in gene_result_list:
      out_file.write("{}\n".format(item))

  # if provided an expanded option
  if file_genetics_expanded_output:
    # generate a file for the regular inputs
    gene_result_list = process_inputs(expanded_list_inputs)

    # write out the results
    with open(file_genetics_expanded_output, "w") as out_file:
      for item in gene_result_list:
        out_file.write("{}\n".format(item))




















