
# imports
import requests
import json
import logging
import sys 
import time
import datetime

# local libraries
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/home/javaprog/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl

# constants
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)

dir_input_json = dir_code + 'MachineLearningPython/DccKP/Translator/Workflows/Json/Queries/'
dir_output_json = "/home/javaprog/Data/Broad/Translator/Testing/Results/"
query_path = ["workflowD2_chem_gene_dis"]
query_path = ["workflowC2_imatnib_ms_gene_mol"]
query_path = ["workflowB1_casey_test"]
query_path = ["workflowD1_dis_named_dis", "workflowD2_chem_gene_dis", "workflowC2_imatnib_ms_gene_mol"]

file_input_pattern_json = "{}{}.json"
# file_aragorn_output_json = dir_json + "Results/workflow1B_aragorn_result.json"
file_output_json = dir_output_json + "{}_{}_{}_result.json"

# constants
url_arax_ara = "https://arax.ncats.io/api/arax/v1.2/query"
url_aragorn_ara = "https://aragorn.renci.org/1.2/query?answer_coalesce_type=all"
map_ara = {'aragorn': url_aragorn_ara}
map_ara = {'arax': url_arax_ara}
map_ara = {'arax': url_arax_ara, 'aragorn': url_aragorn_ara}

# methods

# main
if __name__ == "__main__":

    # loop through ARAs
    for ara_key, url in map_ara.items():
        for path in query_path:
            # get the request payload
            file_input_json = file_input_pattern_json.format(dir_input_json, path)
            with open(file_input_json) as f:
                json_payload = json.load(f)
            logger.info("using json paylload file: {}".format(file_input_json))

            # issue the request
            now = tl.date_to_integer_string()
            start = time.time()
            logger.info("querying ARA: {}".format(url))
            response = requests.post(url, json=json_payload)
            end = time.time()
            time_elapsed = end - start
            logger.info("got ARA result in elapsed time (in seconds) of: {}".format(time_elapsed))

            # convert to json 
            output_json = response.json()

            # find the source counts
            map_count = tl.find_source_tuple_counts(output_json, {"attribute_type_id": "biolink:aggregator_knowledge_source"}, ["attribute_source", "value"], log=True)
            for key, value in map_count.items():
                logger.info("for source: {} got count: {}".format(key, value))

            # save the json
            file_output = file_output_json.format(path ,ara_key, now)
            with open(file_output, 'w') as f:
                json.dump(output_json, f, ensure_ascii=False, indent=4)
            logger.info("wrote out results to file: {}".format(file_output))

            # insert seperator
            logger.info("====================================================")
            logger.info("")

            # print the json output
            # logger.info("output json: \n{}".format(json.dumps(output_json, indent=1)))


    # find how many genetics calls



