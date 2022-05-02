
# imports
import json
import sys 
import logging
import datetime 
import os 
import requests 

# constants
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/home/javaprog/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl
location_servers = dir_code + "MachineLearningPython/DccKP/Translator/Misc/Json/trapiListServices.json"
date_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
location_results = dir_data + "Translator/Workflows/PathwayPpargT2d/AraResults/" + date_now
location_inputs = dir_code + "MachineLearningPython/DccKP/Translator/Workflows/Json/Queries/Pathways/"
file_result = "{}_{}_results.json"
count = 0

# server map
map_servers = {}
map_servers['arax'] = "https://arax.ncats.io/api/arax/v1.2"

# json queries
location_json_pparg_t2d_pathways = location_inputs + "ppargT2dPathwaysQuery.json"

# create the results directory
os.mkdir(location_results)

# read the file
with open(location_json_pparg_t2d_pathways) as file_json: 
    json_input = json.load(file_json)
    query_name = "twohop"

# loop through servers
for server_name, url_trapi in map_servers.items():
    count += 1
    url_query = url_trapi + "/query"
    print("\n{} = got input name: {} to {}".format(count, server_name, url_query))
    response = requests.post(url_query, json=json_input)
    json_output = response.json()
    print("got result: \n{}".format(json_output))

    # add the type of query and service name to the json
    json_output['server_name'] = server_name
    json_output['query_name'] = query_name

    # write out file
    file_output = location_results + "/" + file_result.format(query_name, server_name)
    with open(file_output, 'w') as f:
        print("writing out to file: {}".format(file_output))
        json.dump(json_output, f, ensure_ascii=False, indent=2)


