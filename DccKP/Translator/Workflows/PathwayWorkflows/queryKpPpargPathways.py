
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
location_results = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/Results/" + date_now
location_input_query = dir_code + "MachineLearningPython/DccKP/Translator/Workflows/Json/Queries/Pathways/PpargPathways/{}"
file_result = "{}_{}_results.json"
count = 0
count_max = 300

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
# with open(location_input_query.format("ppargPathwaysReactomeQuery.json")) as file_json: 
#     map_input_json['ppargReactomeLipidDiff'] = json.load(file_json)
# with open(location_input_query.format("ppargPathwaysGo0050872Query.json")) as file_json: 
#     map_input_json['ppargGo0050872LipidDiff'] = json.load(file_json)
# with open(location_input_query.format("ppargPathwaysGo0045444Query.json")) as file_json: 
#     map_input_json['ppargGo0045444LipidDiff'] = json.load(file_json)
# with open(location_input_query.format("ppargPathwaysUmlsQuery.json")) as file_json: 
#     map_input_json['ppargUmlsLipidDiff'] = json.load(file_json)
# with open(location_input_query.format("t2dPathwaysReactomeQuery.json")) as file_json: 
#     map_input_json['t2dReactomeLipidDiff'] = json.load(file_json)
# with open(location_input_query.format("pathwayLipidDiffReactomeQuery.json")) as file_json: 
#     map_input_json['reactomeLipidDiff'] = json.load(file_json)
with open(location_input_query.format("pathwaysReactomeToGoQuery.json")) as file_json: 
    map_input_json['pathwaysReactomeToGo'] = json.load(file_json)

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
            if "chp.thayer.dartmouth.edu" in url_query:
                continue
            print("\n{} = got input name: {} to {}".format(count, name_input, url_query))
            response = requests.post(url_query, json=json_input)

            # try and catch exception
            try:
                json_output = response.json()
                print("got result: \n{}".format(json_output))
            except ValueError:
                print("GOT ERROR: skipping")
                continue


            # add the type of query and service name to the json
            json_output['server_name'] = server_name
            json_output['query_name'] = name_input

            # write out file
            file_output = location_results + "/" + file_result.format(name_input, server_name)
            with open(file_output, 'w') as f:
                print("writing out to file: {}".format(file_output))
                json.dump(json_output, f, ensure_ascii=False, indent=2)
