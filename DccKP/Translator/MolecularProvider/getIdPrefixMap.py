
# imports
import requests
import json

# constants
url_transformers = 'https://translator.broadinstitute.org/molecular_data_provider/transformers'

# methods
def get_prefixes(input_json, log=False):
    ''' method to get the id prefix map for all node types '''
    # initialize
    map_prefix = {}

    # loop through the json
    for item in input_json:
        nodes = item.get('knowledge_map').get('nodes')
        if nodes:
            for key, value in nodes.items():
                if value.get('id_prefixes'):
                    # check if list already built for this node type
                    if map_prefix.get(key):
                        map_prefix.get(key) + value.get('id_prefixes')
                    else:
                        map_prefix[key] = value.get('id_prefixes')

    # return a list of unique values
    for key, value in map_prefix.items():
        map_set = set(map_prefix.get(key))
        map_prefix[key] = list(map_set)

    # return
    return map_prefix


if __name__ == "__main__":
    # call the url and get the json
    response = requests.get(url_transformers)

    # get the json
    output_json = response.json()

    # get the prefix maps
    map_prefix = get_prefixes(output_json)
    for key, values in map_prefix.items():
        print("got key {} and prefixes {}".format(key, values))

