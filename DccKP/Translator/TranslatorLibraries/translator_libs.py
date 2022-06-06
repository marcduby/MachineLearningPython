
# imports
import logging
import requests
import sys
import json
from collections import OrderedDict
import datetime

# constants
logging.basicConfig(level=logging.INFO, format=f'[%(asctime)s] - %(levelname)s - %(name)s : %(message)s')
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)

url_name_search = 'https://name-resolution-sri.renci.org/lookup?string={}'
url_node_normalizer="https://nodenormalization-sri.renci.org/get_normalized_nodes?conflate=true"

# methods
def translate_to_ontology_id(list_input, ontology_prefix, sort_by_ontology=False, log=False):
    '''
    translate array of values using the translator name resolver
    will return multiple rows if multiple results returned for one name
    ex: 
        list_test_result = translate_to_ontology_id(list_test, 'NCBIGene', sort_by_ontology=True)
    get:
        [('MT-ND2', 'NCBIGene:56168'), ('MT-ND2', 'NCBIGene:387315')]
    '''
    # initialize
    list_result = []
    url_name_resolver = "https://name-resolution-sri.renci.org/lookup?string={}"

    # query for the list of names
    for name in list_input:
        url_call = url_name_resolver.format(name)
        try:
            response = requests.post(url_call)
            output_json = response.json()
        except ValueError:
            print("got json error for {}, so skip".format(name))
            continue

        # parse
        for key, value in output_json.items():
            if ontology_prefix in key:
                list_result.append((name, key))

    if sort_by_ontology:
        list_result.sort(key = lambda x: int(x[1].split(":")[1]))

    # return
    return list_result

def get_curie_names(list_curies, log=False):
    ''' method to return list of tuples for ID and the name of the disease '''
    list_result = []
    url = url_node_normalizer + "&curie={}"

    # query
    for curie in list_curies:
        response = requests.get(url.format(curie))
        output_json = response.json()

        # filter
        if output_json.get(curie):
            if output_json.get(curie).get('id'):
                if output_json.get(curie).get('id').get('label'):
                    list_result.append((curie, output_json.get(curie).get('id').get('label')))

    # return
    return list_result

def get_curie_names_post(list_curies, log=False):
    ''' method to return list of tuples for ID and the name of the disease '''
    list_result = []

    # query
    payload = {"curies": list_curies}
    response = requests.post(url_node_normalizer, json=payload)
    output_json = response.json()

    # filter
    for key, value in output_json.items():
        if value is None:
            list_result.append((key, value))
        else:
            list_result.append((key, value.get("id").get("label")))
                    
    # return
    return list_result

def find_ontology(name, list_ontology, debug=False):
    '''will call REST api and will return ontology id if name exact match and ontology prefix in list provided '''
    # initialize
    ontology_id = None

    # call the url
    response = requests.post(url_name_search.format(name.replace("-", " ")))
    output_json = response.json()

    # loop through results, find first exact result
    for key, values in output_json.items():
        # print("key: {}".format(key))
        # print("value: {}\n".format(values))
        # do MONDO search first since easiest comparison
        for item in list_ontology:
            if item in key:
                if name.lower() in map(str.lower, values):
                    ontology_id = key
                    break

    # log
    if debug:
        logger.info("for: {} found: {}".format(name, ontology_id))

    # return
    return ontology_id

def get_nodes_one_hop(url, list_source, list_target, list_source_categories, list_target_categories, list_predicates, log=False):
    ''' method to query a trapi url and get the resulting node list back '''
    list_result = []

    # query
    json_response = query_one_hop(url, list_source, list_target, list_source_categories, list_target_categories, list_predicates, log)

    # loop and build the list
    list_nodes = json_response.get("message").get("knowledge_graph").get("nodes")
    if list_nodes and len(list_nodes) > 1:
        for key, value in list_nodes.items():
            list_result.append((key, value.get("name")))

    # log
    if log:
        logger.info("got {} resulting nodes: {}".format(len(list_result), list_result))

    # return
    return list_result

def query_one_hop(url, list_source, list_target, list_source_categories, list_target_categories, list_predicates, log=False):
    ''' method to call a trapi url '''
    response = None

    # build the payload
    payload = build_one_hop_payload(list_source, list_target, list_source_categories, list_target_categories, list_predicates, log=log)

    # call the url
    logger.info("query: {}".format(url))
    response = requests.post(url, json=payload)
    output_json = response.json()
    logger.info("got results from: {}".format(url))

    # log
    # if log:
    #     logger.info("got response: {}".format(output_json))

    # return the json
    return output_json

def build_one_hop_payload(list_source, list_target, list_source_categories, list_target_categories, list_predicates, log=False):
    ''' method to build a one hop json payload for a trapi query '''
    payload = {}

    # build the payload
    nodes = {"n00": build_trapi_query_node(list_source, list_source_categories, log=True), "n01": build_trapi_query_node(list_target, list_target_categories, log=True)}
    edge = {"subject": "n00", "object": "n01"}
    if list_predicates and len(list_predicates) > 0:
        edge["predicates"]= list_predicates
    edges = {"e00": edge}
    payload["message"] = {"query_graph": {"edges": edges, "nodes": nodes}}

    # log
    if log:
        logger.info("build trapi payload: \n{}".format(json.dumps(payload, indent=4)))

    # return
    return payload

def build_trapi_query_node(list_source, list_source_categories, log=False):
    ''' method to build a trapi query node '''
    node = {}

    # log
    # if log:
    #     logger.info("got id: {} and categories: {}".format(list_source, list_source_categories))

    # build the node
    if list_source and len(list_source) > 0:
        node['ids'] = list_source
    if list_source_categories and len(list_source_categories) > 0:
        node['categories'] = list_source_categories

    # return
    return node

def recursively_find_source_tuples(input_object, type_map, list_elements, level=0, log=False):
    ''' 
    recursively go through map to find data of type given and pull list elements in tuples 
    '''
    # initialize
    list_result = []

    # log
    # if log:
    #     logger.info("got recursed input object of size: {} for level: {}".format(len(input_object), level))

    # check if list
    if isinstance(input_object, list):
        for item in input_object:
            list_temp = recursively_find_source_tuples(item, type_map, list_elements, level=level+1, log=log)
            list_result = list_result + list_temp

    # check if map, then recurse over children maps
    elif isinstance(input_object, dict):
        for key, value in input_object.items():
            # logger.info("key: {}".format(key))
            if isinstance(value, dict) or isinstance(value, list):
                list_temp = recursively_find_source_tuples(value, type_map, list_elements, level=level+1, log=log)
                list_result = list_result + list_temp

        # add to results if keys all have list_elements
        # logger.info("got key list: {}".format(input_object.keys()))
        # if "attributes" in list(input_object.keys()): 
        #     logger.info("got key list: {} for level: {}".format(input_object.keys(), level))
        # else:
        #     logger.info("got key list: {}".format(input_object.keys()))

        if set(list_elements).issubset(input_object.keys()):
            if type_map:
                addInList = True
                for type_key, type_value in type_map.items():
                    if input_object.get(type_key) != type_value:
                        addInList = False
                        break

                if addInList:
                    list_result.append([input_object.get(item_key) for item_key in list_elements])
            else:
                list_result.append([input_object.get(item_key) for item_key in list_elements])

    # log
    # if log:
    #     logger.info("got recursed list of size: {}".format(len(list_result)))

    # return
    return list_result

def find_source_tuple_counts(input_object, type_map, list_elements, log=False):
    ''' 
    recursively go through map to find data of type given and pull list elements in tuples 
    '''
    # initialize
    map_result = {}
    list_result = []

    # look for message/knowledge_graph/edges
    if input_object and input_object.get("message") and input_object.get("message").get("knowledge_graph") and input_object.get("message").get("knowledge_graph").get("edges"):
        # get the recursed list
        list_result = recursively_find_source_tuples(input_object.get("message").get("knowledge_graph").get("edges"), type_map, list_elements, level=0, log=log)

        # loop and count
        for item in list_result:
            # logger.info("got result: {}".format(item))
            # create map key
            key = ""
            for temp in item:
                key = key + " - " + str(temp) 

            # increment for key
            if not map_result.get(key):
                map_result[key] = 1
            else :
                map_result[key] = map_result.get(key) + 1

        # order by result
        map_result = OrderedDict(sorted(map_result.items()))

    else:
        logger.error("found no message/knowdledge_graph/edges element")

    # log
    if log:
        logger.info("got ordered map of size: {}".format(len(map_result)))

    # return
    return map_result

def date_to_integer_string(dt_time=None, log=True):
    ''' return date into string version '''
    now = dt_time
    if not dt_time:
        now = datetime.datetime.now()

    if log:
        logger.info("got time input: {} so using: {}".format(dt_time, now))

    # return
    return str(int(now.year * 1e8 + now.month * 1e6 + now.day* 1e4 + now.hour* 1e2 + now.minute))

def get_trapi_servers(json_servers, type='KP', log=True):
    '''
    query the smart api and get the servers of type specified
    '''
    # initialize
    map_servers = {}

    # get the kps
    for entry in json_servers.get('hits'):
        if entry.get('info').get('x-translator').get('component'):
            map_server = {'comp': entry.get('info').get('x-translator').get('component')}
            if map_server.get('comp') == type:
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

    # return
    return map_servers.values()


def get_trapi_aras(json_servers, log=True):
    '''
    query the smart api and get the ARAs
    '''
    # initialize
    map_servers = get_trapi_servers(json_servers, 'ARA', log=log)

    # return
    return map_servers

def get_trapi_kps(json_servers, log=True):
    '''
    query the smart api and get the KPs
    '''
    # initialize
    map_servers = get_trapi_servers(json_servers, 'KP', log=log)

    # return
    return map_servers

def is_string_in(string_search, element, log=False):
    ''' 
    find the string if in the tree structure; return true if found
    '''
    # initialize
    is_contained = False

    # log
    if log:
        print("looking for string: {}".format(string_search))    

    # loop of dict or list
    if isinstance(element, list):
        for child in element:
            if is_string_in(string_search, child):
                return True
    elif isinstance(element, dict):
        for child in element.values():
            if is_string_in(string_search, child):
                return True
    else:
        is_contained = string_search in str(element)

    # return
    return is_contained


def find_all_instances_string(string_search, element, log=False):
    ''' 
    find all the instances of the string if in the tree structure; return list
    '''
    # initialize
    list_result = []

    # log
    if log:
        print("looking for string: {}".format(string_search))    

    # loop of dict or list
    if isinstance(element, list):
        for child in element:
            list_result += find_all_instances_string(string_search, child, log)

    elif isinstance(element, dict):
        for child in element.values():
            list_result += find_all_instances_string(string_search, child, log)
                
    else:
        if string_search in str(element):
            list_result.append(element)


    # return
    return list_result

def count_trapi_results_edges(json_result, log=False):
    '''
    will take a trapi json result file and return the result edges in the graph
    '''
    count = 0

    if json_result.get('message'):
        if json_result.get('message').get('knowledge_graph'):
            if json_result.get('message').get('knowledge_graph').get('edges'):
                count = len(json_result.get('message').get('knowledge_graph').get('edges'))

    # log
    if log:
        print("found {} edge results".format(count))

    # return
    return count


if __name__ == "__main__":
    name_test = "PTPA"
    curie_id = find_ontology(name_test, 'NCBIGene', debug=True)
    logger.info("for: {} found: {}".format(name_test, curie_id))

    # test the trapi calling
    list_subject = ["MESH:D056487", "MONDO:0005359", "SNOMEDCT:197358007", "MESH:D056487", "NCIT:C26991"]
    list_categories = ["biolink:DiseaseOrPhenotypicFeature"]
    list_predicates = ["biolink:has_real_world_evidence_of_association_with"]
    build_one_hop_payload(list_subject, None, list_categories, list_categories, list_predicates, log=True)

    # get the nodes
    # url_arax = "https://arax.ncats.io/api/arax/v1.1/query"
    # url_aragorn = "https://aragorn.renci.org/1.1/query?answer_coalesce_type=all"
    # for url in [url_arax, url_aragorn]:
    #     list_nodes = get_nodes_one_hop(url, list_subject, None, list_categories, list_categories, list_predicates, log=True)
    #     for item in list_nodes:
    #         logger.info("got node: {}".format(item))
    #     logger.info("for {}, got list of size: {}".format(url, len(list_nodes)))

    # get datetime
    now = datetime.datetime.now()
    logger.info("got date time: {}".format(date_to_integer_string(now)))

    # test batch normalization
    list_test = ["CHEBI:59683",
                    "CHEMBL.COMPOUND:CHEMBL2108558",
                    "LOINC:55275-2",
                    "MESH:C040391",
                    "MESH:C064613",
                    "MESH:C069356",
                    "MESH:C071458",
                    "MESH:C093154",
                    "MESH:C110500",
                    "MESH:C115528",
                    "MESH:C518324",
                    "MESH:C519298",
                    "MESH:D000071020",
                    "MESH:D006679",
                    "MESH:D016031",
                    "MESH:D053218",
                    "MONDO:0000775",
                    "MONDO:0001475",
                    "MONDO:0002184",
                    "MONDO:0004335",
                    "MONDO:0005071",
                    "MONDO:0005267",
                    "MONDO:0005354",
                    "MONDO:0005359",
                    "MONDO:0005366",
                    "MONDO:0005790",
                    "MONDO:0007745",
                    "MONDO:0013209",
                    "MONDO:0013282",
                    "MONDO:0013433",
                    "MONDO:0018229",
                    "MONDO:0043693",
                    "NCIT:C29933",
                    "OMIM:MTHU002997",
                    "OMIM:MTHU012757",
                    "OMIM:MTHU013583",
                    "OMIM:MTHU021860",
                    "OMIM:MTHU030933",
                    "OMIM:MTHU034288",
                    "OMIM:MTHU045723",
                    "OMIM:MTHU048033",
                    "OMIM:MTHU048989",
                    "SCITD:143472004",
                    "SCITD:406104003",
                    "SCITD:86259008",
                    "SCTID:207471009",
                    "SCTID:64411004",
                    "UMLS:C0149709",
                    "UMLS:C0262505",
                    "UMLS:C0342271",
                    "UMLS:C0455417",
                    "UMLS:C0455540",
                    "UMLS:C0473117",
                    "UMLS:C0552479",
                    "UMLS:C0948251",
                    "UMLS:C4049267",
                    "UMLS:C4554323"]
    list_test = ['UMLS:C1720947', 'NCIT:C3143', 'DOID:863', 'EFO:0005556', 'MESH:D004066', 'UMLS:C0671077', 'MESH:C518324', 'DOID:114', 'UMLS:C3277286', 'UMLS:C0012242', 'UMLS:C1840547', 'MESH:D009422', 'HP:0410323', 'HP:0000952', 'UMLS:C3241919', 'UMLS:C0003417', 'UMLS:C1565321', 'EFO:0009482', 'UMLS:C0559031', 'UMLS:C0038325', 'MONDO:0007745', 'NCIT:C143255', 'UMLS:C1857414', 'MESH:C519298', 'MESH:D006679', 'HP:0011024', 'UMLS:C0022346', 'MESH:D013262', 'MONDO:0000775', 'DOID:0060500', 'MONDO:0001475', 'MONDO:0043693', 'UMLS:C0027765', 'NCIT:C3385', 'UMLS:C4231138', 'NCIT:C2990', 'DOID:0050426', 'UMLS:C0948251', 'UMLS:C3658302', 'MONDO:0004335', 'EFO:1001248', 'MONDO:0005359', 'UMLS:C0759708', 'MESH:D000982', 'UMLS:C0017551', 'MESH:D065626', 'UMLS:C0524912', 'UMLS:C0017178', 'UMLS:C1442981', 'EFO:0004220', 'UMLS:C0342271', 'UMLS:C4277647', 'NCIT:C84427', 'UMLS:C0853697', 'MESH:D019896', 'UMLS:C3658301', 'UMLS:C0013182', 'DOID:12549', 'MESH:C071458', 'DOID:2739', 'MESH:C040391', 'MESH:D004342', 'NCIT:C3079', 'MESH:D005878', 'UMLS:C0023896', 'UMLS:C1856453', 'MESH:D007565', 'UMLS:C0018799', 'UMLS:C1837818', 'EFO:0000618', 'NCIT:C29448', 'MESH:D000081226', 'MESH:D053218', 'UMLS:C2608081', 'UMLS:C0455540', 'UMLS:C0141982', 'NCIT:C34783', 'MESH:D019694', 'MESH:D056487', 'MESH:D000071020', 'UMLS:C4277682', 'MONDO:0013209', 'UMLS:C0233523', 'NCIT:C79484', 'UMLS:C0455417', 'NCIT:C84444', 'UMLS:C4016206', 'UMLS:C0566602', 'NCIT:C80520', 'MONDO:0044719', 'UMLS:C3278891', 'DOID:0080546', 'MONDO:0005071', 'EFO:1000905', 'UMLS:C0646266', 'UMLS:C5139486', 'UMLS:C0081424', 'MESH:C110500', 'MONDO:0005790', 'UMLS:C0648354', 'DOID:77', 'NCIT:C35299', 'MESH:C064613', 'MESH:D019698', 'UMLS:C0027947', 'UMLS:C0473117', 'EFO:0008573', 'MESH:D005767', 'MESH:D006331', 'UMLS:C1262760', 'UMLS:C0524909', 'UMLS:C1274933', 'UMLS:C3276783', 'NCIT:C81229', 'UMLS:C0221757', 'NCIT:C26835', 'MESH:D008108', 'UMLS:C1840548', 'NCIT:C84397', 'UMLS:C0141981', 'UMLS:C4554323', 'UMLS:C4505492', 'NCIT:C29933', 'MONDO:0013433', 'EFO:0004228', 'UMLS:C4049267', 'MONDO:0005267', 'MESH:D016031', 'UMLS:C0019699', 'EFO:0000405', 'UMLS:C1429314', 'EFO:0003777', 'UMLS:C4023588', 'HP:0001875', 'NCIT:C15271', 'EFO:0003095', 'NCIT:C15175', 'UMLS:C4505493', 'UMLS:C0019159', 'UMLS:C2717837', 'UMLS:C2010848', 'MESH:C069356', 'UMLS:C1956568', 'MONDO:0018229', 'DOID:13372', 'NCIT:C84729', 'UMLS:C2674487', 'MESH:D056486', 'UMLS:C0149709', 'UMLS:C0262505', 'MESH:C093154', 'UMLS:C4279912', 'UMLS:C1870209', 'UMLS:C0019193', 'UMLS:C0400966', 'UMLS:C1969756', 'UMLS:C3658290', 'DOID:1227', 'EFO:0004239', 'MESH:D006506', 'MONDO:0005354', 'UMLS:C0860207', 'DOID:2044', 'DOID:0080208', 'MESH:D029846', 'NCIT:C3096', 'EFO:0004276', 'UMLS:C2750833', 'MESH:D009503', 'MONDO:0013282', 'UMLS:C0524910', 'MONDO:0005366', 'DOID:0060643', 'NCIT:C94296', 'UMLS:C0242183', 'EFO:0007305', 'UMLS:C0552479', 'MESH:C115528', 'MONDO:0002184']
    list_result = get_curie_names_post(list_test)
    for (curie, name) in list_result:
        logger.info("got id: {} and name: {}".format(curie, name))



