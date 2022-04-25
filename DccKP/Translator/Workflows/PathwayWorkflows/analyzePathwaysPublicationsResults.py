
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
location_results = dir_data + "Translator/Workflows/PathwayPpargT2d/Results/" + date_now
location_inputs = dir_code + "MachineLearningPython/DccKP/Translator/Workflows/Json/Queries/Pathways/"
file_result = "{}_{}_results.json"

# initialize
count = 0
count_max = 500

# json queries
location_json_pparg_pathways = location_inputs + "ppargPathwaysQuery.json"
location_json_t2d_pathways = location_inputs + "t2dPathwaysQuery.json"
location_json_pparg_t2d = location_inputs + "t2dPpargQuery.json"

# read the file
with open(location_servers) as file_json: 
    json_servers = json.load(file_json)

# load the trapi kps
list_servers = tl.get_trapi_kps(json_servers)
print("got {} KPs".format(len(list_servers)))
print("datetime: {}".format(date_now))

# create the results directory
os.mkdir(location_results)

# load the json inputs
map_input_json = {}
with open(location_json_pparg_pathways) as file_json: 
    map_input_json['ppargPathways'] = json.load(file_json)
with open(location_json_t2d_pathways) as file_json: 
    map_input_json['t2dPathways'] = json.load(file_json)
with open(location_json_pparg_t2d) as file_json: 
    map_input_json['t2dPparg'] = json.load(file_json)

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
            json_output = response.json()
            print("got result: \n{}".format(json_output))

            # add the type of query and service name to the json
            json_output['server_name'] = server_name
            json_output['query_name'] = name_input

            # write out file
            file_output = location_results + "/" + file_result.format(name_input, server_name)
            with open(file_output, 'w') as f:
                print("writing out to file: {}".format(file_output))
                json.dump(json_output, f, ensure_ascii=False, indent=2)




