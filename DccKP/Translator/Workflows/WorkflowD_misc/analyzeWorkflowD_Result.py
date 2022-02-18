
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


# main
if __name__ == "__main__":
    # open the input file
    with open(file_input_json) as f:
        json_payload = json.load(f)
        logger.info("read in file: {}".format(file_input_json))

    # recursively count the results
    map_count = tl.find_source_tuple_counts(json_payload, {"attribute_type_id": "biolink:aggregator_knowledge_source"}, ["attribute_source", "value"], log=True)
    for key, value in map_count.items():
        logger.info("for source: {} got count: {}".format(key, value))

