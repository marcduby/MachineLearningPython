
import requests
import json
import logging
import sys 
import time
import datetime

# local libraries
dir_code = "/home/javaprog/Code/PythonWorkspace/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl

# constants
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)

# file constants
dir_data = "/home/javaprog/Data/Broad/Translator/Testing/Results/"
file_input_json = dir_data + "workflowD1_dis_named_dis_arax_result.json"
file_input_json = "/home/javaprog/Data/Broad/Translator/Testing/Results/dccWorkflowT2dDiseaseDrug_arax_202110052210_result.json"
file_input_json = "/home/javaprog/Data/Broad/Translator/Testing/Results/dccWorkflowT2dDiseaseDrug_arax_202110052248_result.json"

# main
if __name__ == "__main__":
    # open the input file
    with open(file_input_json) as f:
        json_payload = json.load(f)
        logger.info("read in file: {}".format(file_input_json))

    # recursively count the results
    map_nodes = {}
    for key, value in json_payload.get("message", {}).get("knowledge_graph", {}).get("nodes", {}).items():
        map_nodes[key] = value.get("name")

    for key, value in map_nodes.items():
        if ("MONDO" not in key):
            print("got drug: {} with id: {}".format(value, key))
    print("")
    for key, value in map_nodes.items():
        if ("MONDO" in key):
            print("got disease: {} with id: {}".format(value, key))


