
# imports
import requests
import json
import logging
import sys 
import time

# local libraries
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/home/javaprog/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl

# constants
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)

dir_json = dir_code + 'MachineLearningPython/DccKP/Translator/Workflows/Json/'
file_input_json = dir_json + "Queries/Pathways/ppargT2dPathwaysQuery.json"
# file_aragorn_output_json = dir_json + "Results/workflow1B_aragorn_result.json"
file_output_json = dir_json + "Results/ppargT2dPathways_{}_result.json"

# constants
url_arax_ara = "https://arax.ncats.io/api/arax/v1.2/query"
# url_aragorn_ara = "https://aragorn.renci.org/1.2/query?answer_coalesce_type=all"
url_aragorn_ara = "https://aragorn.renci.org/1.2/query?answer_coalesce_type=all"
map_ara = {'arax': url_arax_ara, 'aragorn': url_aragorn_ara}
map_ara = {'arax': url_arax_ara}
map_ara = {'aragorn': url_aragorn_ara}

# methods

# main
if __name__ == "__main__":
    # get the request payload
    with open(file_input_json) as f:
        json_payload = json.load(f)

    # loop through ARAs
    for ara_key, url in map_ara.items():
        # issue the request
        start = time.time()
        logger.info("querying ARA: {}".format(url))
        logger.info("querying with json: \n{}".format(json_payload))
        response = requests.post(url, json=json_payload)
        end = time.time()
        time_elapsed = end - start
        logger.info("got ARA result in elapsed time (in seconds) of: {}".format(time_elapsed))

        # convert to json 
        output_json = response.json()

        # save the json
        file_output = file_output_json.format(ara_key)
        with open(file_output, 'w') as f:
            json.dump(output_json, f, ensure_ascii=False, indent=4)
        logger.info("wrote out results to file: {}".format(file_output))

        # print the json output
        logger.info("output json: \n{}".format(json.dumps(output_json, indent=1)))

    # find how many genetics calls



