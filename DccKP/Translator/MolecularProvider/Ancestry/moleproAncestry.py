

# imports
import requests

# constants
ANCESTRY_PREFIX = ['MONDO']

# methods
def make_node_producer_payload(list_ids, debug=False):
    '''
    method to make a payload for the node producer translator post call
    '''
    # build the control list
    list_control = []
    for id in list_ids:
        list_control.append({'name': 'id', 'value': id})

    # build the map
    result = {'name': 'MoleProDB node producer', 'controls': list_control}

    # log
    if debug:
        print("got payload: {}".format(result))

    # return
    return result

def get_node_producer_collection_id(query_url, list_ids, debug=False):
    '''
    method tto call the node producer transformer with the proper payload
    '''
    # create the payload
    query_map = make_node_producer_payload(list_ids, debug)

    # call the REST POST call and get the response
    response = requests.post(query_url, json=query_map).json()

    # log
    if debug:
        print("got node producer results: \n{}".format(response))

    # parse the results
    result_id = response.get('id')

    # return
    return result_id

def get_node_ancestry_url(query_url, collection_id, debug=False):
    '''
    method to call the molepro hierarchy with the given collection id
    '''
    # make the payload
    controls_list = [{'name': 'name_source', 'value': 'MolePro'}, {'name': 'element_attribute', 'value': 'biolink:publication'}]
    query_map = {'name': 'MoleProDB hierarchy transformer', 'collection_id': collection_id, 'controls': controls_list}

    # call the REST POST
    response = requests.post(query_url, json=query_map).json()

    # get the result
    result_url = response.get('url')

    # log
    if debug:
        print("got hierarchy url: {}".format(result_url))

    # return
    return result_url

def get_node_ancestry_from_url(query_url, debug=False):
    '''
    method to call the collection specific url and pull out the ancestry nodes
    '''
    # initialize
    result_map = {}

    # call the REST GET endpoint
    response = requests.get(query_url).json()

    # loop and pull out results
    elements_list = response.get('elements')
    if elements_list:
        for element in elements_list:
            # get the child id
            child_id = element.get('id')

            # loop through the connections
            connections = element.get('connections')
            if connections:
                for connection in connections:
                    parent_id = connection.get('source_element_id')

                    # add to the result map
                    if not result_map.get(parent_id):
                        result_map[parent_id] = []
                    result_map.get(parent_id).append(child_id)
                    

    # log
    if debug:
        print("got ancestry result map: {}".format(result_map))

    #return
    return result_map

def get_ancestry_map(query_url, list_ids, debug=False):
    '''
    get the ancestry map from molepro given an input list
    key is an item in the list
    value is the list of ancestors/translated IDs
    '''
    # make sure IDs are unique
    list_ids = list(set(list_ids))

    # split based on prefix of ID
    list_temp = []
    map_skipped = {}
    for item in list_ids:
        if item.split(":")[0] in ANCESTRY_PREFIX:
            list_temp.append(item)
        else:
            map_skipped[item] = item

    # log
    if debug:
        print("got skipped map: {}".format(map_skipped))

    # get the collection id
    collection_id = get_node_producer_collection_id(query_url, list_temp, debug)

    # get the query get url
    get_url = get_node_ancestry_url(query_url, collection_id, debug)

    # get the ancestry map
    map_ancestry = get_node_ancestry_from_url(get_url, debug)

    # log
    if debug:
        print("got skipped map: {}".format(map_skipped))
        print("got searched map: {}".format(map_ancestry))

    # combine the skipped items with the searched items
    if len(map_skipped) > 0:
        if debug:
            print("merging maps")
        map_ancestry = dict(map_ancestry, **map_skipped)
        
    # log
    if debug:
        print("got result map: {}".format(map_ancestry))

    # return
    return map_ancestry

if __name__ == "__main__":
    # test payload creation
    list_ids = ['MONDO:0014488', 'MONDO:0005148', 'MONDO:0011057']
    make_node_producer_payload(list_ids, debug=True)

    # test node producer
    query_url = "https://translator.broadinstitute.org/molecular_data_provider/transform"
    test_collection_id = get_node_producer_collection_id(query_url, list_ids, True)
    print("got node producer collection id: {}".format(test_collection_id))

    # test the hierarchy producer
    test_url = get_node_ancestry_url(query_url, test_collection_id, True)

    # test getting the ancestry map
    test_ancestry_map = get_node_ancestry_from_url(test_url, True)
    for key in test_ancestry_map.keys():
        print("for {} got child list of size {}".format(key, len(test_ancestry_map.get(key))))

    # test getting the ancestry map in one call
    print("\n\n_______________________________\n\n")
    test_ancestry_map = get_ancestry_map(query_url, list_ids, True)
    for key in test_ancestry_map.keys():
        print("for {} got child list of size {}".format(key, len(test_ancestry_map.get(key))))

    # test skipping
    print("\n\n_______________________________\n\n")
    list_ids = ['MONDO:0014488', 'GENE:0005']
    test_ancestry_map = get_ancestry_map(query_url, list_ids, True)
    for key in test_ancestry_map.keys():
        print("for {} got child list of size {}".format(key, len(test_ancestry_map.get(key))))

# notes: 
# add query id into it's own list
# only call disease/phenotype - know by prefix
