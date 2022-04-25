# imports
import logging
import requests
import sys
import json
import time


# constants
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/home/javaprog/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl

# constants
location_servers = dir_code + "MachineLearningPython/DccKP/Translator/Misc/Json/trapiListServices.json"

# read the file
with open(location_servers) as file_json: 
    json_servers = json.load(file_json)

# get the services
map_servers = {}

print("got {} number of services".format(len(json_servers.get('hits'))))
for entry in json_servers.get('hits'):
    if entry.get('info').get('x-translator').get('component'):
        map_server = {'comp': entry.get('info').get('x-translator').get('component')}
        if map_server.get('comp') == 'KP':
            if entry.get('info').get('x-translator').get('infores'):
                map_server['info'] = entry.get('info').get('x-translator').get('infores')
                if entry.get('servers'):
                    for serv in entry.get('servers'):
                        if serv.get('x-maturity') == 'production':
                            if serv.get('url'):
                                map_server['url'] = serv.get('url')
                                if entry.get('info').get('x-trapi'):
                                    map_server['version'] = entry.get('info').get('x-trapi').get('version')
                                    map_servers[map_server.get('info')] = map_server


# print results
print("got {} number of KPs".format(len(map_servers)))
for item in map_servers.values():
    print("got trapi: {}".format(item))

