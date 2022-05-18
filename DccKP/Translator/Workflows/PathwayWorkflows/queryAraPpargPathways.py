
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
location_results = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/AraResults/" + date_now
location_input_query = dir_code + "MachineLearningPython/DccKP/Translator/Workflows/Json/Queries/Pathways/PpargPathways/{}"
file_result = "{}_{}_results.json"
count = 0
count_max = 500

# server map
map_servers = {}
map_servers['arax'] = "https://arax.ncats.io/api/arax/v1.2"

# json queries
map_input_json = {}
with open(location_input_query.format("ppargT2dPathwayGo0045444Query.json")) as file_json: 
    map_input_json['ppargT2dPathwayGo0045444Query'] = json.load(file_json)
# with open(location_input_query.format("ppargT2dPathwayGo0050872Query.json")) as file_json: 
#     map_input_json['ppargT2dPathwayGo0050872Query'] = json.load(file_json)
with open(location_input_query.format("ppargT2dPathwaysQuery.json")) as file_json: 
    map_input_json['ppargT2dPathwaysQuery'] = json.load(file_json)

# create the results directory
os.mkdir(location_results)

# loop through servers
for server_name, server_url in map_servers.items():
    # increment count
    count += 1

    # loop through input queries
    if count < count_max:
        # loop through input queries
        for name_input, json_input in map_input_json.items():
            url_query = server_url + "/query"
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


