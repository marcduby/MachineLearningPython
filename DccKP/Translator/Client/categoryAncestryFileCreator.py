
# imports
import json
import requests 
from contextlib import closing

# constants
file_prepend = "/Users/mduby"
file_prepend = "/home/javaprog"
url_node_normalizer = "https://bl-lookup-sri.renci.org/bl/{}/ancestors?version={}"
url_molepro_predicates = 'https://translator.broadinstitute.org/molepro/trapi/v1.0/predicates'
url_genepro_predicates = 'https://translator.broadinstitute.org/genetics_provider/trapi/v1.0/predicates'
file_molepro = file_prepend + '/Data/Broad/Translator/Molepro/biolinkAncestry.json'
file_genepro = file_prepend + '/Data/Broad/Translator/GeneticsPro/biolinkAncestry.json'
file_query = file_prepend + '/Data/Broad/Translator/Client/afibGeneRelated.json'
file_chem_query = file_prepend + '/Data/Broad/Translator/Client/chemGeneRelated.json'
file_blank_query = file_prepend + '/Data/Broad/Translator/Client/blankGeneRelated.json'


def get_biolink_ancestors(entity_name, api_version='latest'):
    ''' retrieve the ancestors of a entity type '''
    ancestors = []

    # build the url
    query_url = url_node_normalizer.format(entity_name, api_version)

    # query the url
    print("finding ancestors for {}".format(entity_name))
    with closing(requests.get(query_url)) as response_obj:
        if response_obj is not None and response_obj.status_code != 404:
            ancestors = response_obj.json()

    # return list
    return ancestors

def get_entities_from_predicates(predicate_url):
    ''' returns a list of objects defined in a predicate (entities, categories, predicates) '''
    entity_set = set()
    response = None

    # query the url
    with closing(requests.get(predicate_url)) as response_obj:
        response = response_obj.json()

    if response is not None:
        # get all the types
        key_list = list(response.keys())
        entity_set.update(key_list)

        # get all the categories
        for item in key_list:
            category_list = list(response.get(item).keys())
            entity_set.update(category_list)

            # get all the predicates
            for category in category_list:
                entity_set.update(response.get(item).get(category))

    # rerturn
    return list(entity_set)

def build_ancestry_map(predicate_url):
    ''' build a map of biolink terms to predicate term list based on predicate url '''
    ancestry_map = {"all": []}

    # get the entities
    type_list = get_entities_from_predicates(predicate_url)

    # add all entities as child of 'all'
    for item in type_list:
        ancestry_map.get('all').append(item)

    # for each entity, get their ancestors
    for item in type_list:
        # add itself to the map
        if ancestry_map.get(item) is None:
            ancestry_map[item] = []
        ancestry_map.get(item).append(item)
        
        item_ancestry_list = get_biolink_ancestors(item)

        for ancestor in item_ancestry_list:
            # add them to each ancestors list
            if ancestry_map.get(ancestor) is None:
                ancestry_map[ancestor] = []
            ancestry_map.get(ancestor).append(item)

    # return
    return ancestry_map

def create_query_string(subject_type=None, object_type=None, predicate=None):
    ''' creates representative string from query objects '''
    query = ""

    for item in [subject_type, predicate, object_type]:
        # print("adding {}".format(item))
        if item is None:
            query += "all "
        else:
            query += item + " "

    # return
    return query    

def create_predicate_query_list(predicate_json):
    ''' create a list query strings of all accepted queries for the given predicate '''
    query_list = []

    # go through json and build all possible queries
    if predicate_json is not None:
        # get all the subject categories
        subject_category_list = list(response.keys())

        # get all the object categories
        for subject_category in subject_category_list:
            object_category_list = list(response.get(subject_category).keys())

            # get all the predicates
            for object_category in object_category_list:
                for predicate in response.get(subject_category).get(object_category):
                    query_list.append(create_query_string(subject_category, object_category, predicate))

    # return
    return query_list

def build_query_descendant_list(descendant_map, subject_type, object_type, predicate):
    ''' returns all the possible descendant queries in string format '''
    query_list = []

    # check the inputs
    if subject_type is None or len(subject_type) < 1:
        subject_type = ['all']
    if object_type is None or len(object_type) < 1:
        object_type = ['all']
    if predicate is None or len(predicate) < 1:
        predicate = ['all']
    
    # make arrays
    subject_type = make_into_array(subject_type)
    object_type = make_into_array(object_type)
    predicate = make_into_array(predicate)

    # loop
    for sub in subject_type:
        if descendant_map.get(sub):
            for dsub in descendant_map.get(sub):
                for pred in predicate:
                    if descendant_map.get(pred):
                        for dpred in descendant_map.get(pred):
                            for obj in object_type:
                                if descendant_map.get(obj):
                                    for dobj in descendant_map.get(obj):
                                        query_list.append(create_query_string(dsub, dobj, dpred))

    # return
    return query_list

def get_query_parts(query_json):
    ''' returns subject/object/predicate of the json query '''
    sub = ['all']
    obj = ['all']
    pred = ['all']
    subject_key = None
    object_key = None

    # get the objects
    qg = query_json.get('message').get('query_graph')
    # print("got {}".format(qg))
    if len(qg.get('edges').keys()) > 0:
        for key in list(qg.get('edges').keys()):
            if qg.get('edges').get(key).get('predicate'):
                pred = qg.get('edges').get(key).get('predicate')
                subject_key = qg.get('edges').get(key).get('subject')
                object_key = qg.get('edges').get(key).get('object')
    if subject_key:
        if qg.get('nodes').get(subject_key).get('category'):
            sub = qg.get('nodes').get(subject_key).get('category')
    if object_key:
        if qg.get('nodes').get(object_key).get('category'):
            obj = qg.get('nodes').get(object_key).get('category')

    # return
    return sub, obj, pred 

def get_all_overap_queries(ancestor_map, predicate_json, query_json):
    ''' returns the intersection of the predicates and expanded query list '''
    # get the parts of the query
    sub, obj, pred = get_query_parts(query_json)

    # get the intersection
    result = get_all_overap_queries_for_parts(ancestor_map, predicate_json, sub, obj, pred)

    # return
    return result

def get_all_overap_queries_for_parts(ancestor_map, predicate_json, subject_type, object_type, predicate):
    ''' returns the intersection of the predicates and expanded parts from query '''
    predicate_list =  create_predicate_query_list(predicate_json)
    # print("predicate {}".format(len(predicate_list)))
    query_list = build_query_descendant_list(ancestor_map, subject_type, object_type, predicate)
    # print("query {}".format(len(query_list)))

    # get the intersection
    result = list(set(predicate_list) & set(query_list))
    # print("result {}".format(len(result)))

    # return
    return result

def make_into_array(temp):
    ''' turns input into array if not already '''
    result = temp

    # checks
    if temp is None:
        result = []
    if type(temp) != type([]):
         result = [temp]

    # return
    return result

if __name__ == "__main__":
    # get the gene ancestors
    gene_list = get_biolink_ancestors('Gene')
    for item in gene_list:
        print("got ancestor {}".format(item))

    print()

    # get the molepro predicate objects
    print("testing entity list from predicate")
    entity_list = get_entities_from_predicates(url_molepro_predicates)
    for item in entity_list:
        print("got entity: {}".format(item))

    print()

    # get the map of ancestors for molepro
    print("testing ancestor map from predicate")
    # ancestor_map = build_ancestry_map(url_molepro_predicates)
    # for key in list(ancestor_map.keys()):
    #     print("for ancestor {} got list {}".format(key, ancestor_map.get(key)))
    ancestor_map = build_ancestry_map(url_genepro_predicates)
    for key in list(ancestor_map.keys()):
        print("for ancestor {} got list {}".format(key, ancestor_map.get(key)))
    # file_to_write = file_molepro
    file_to_write = file_genepro
    with open(file_to_write, 'w') as json_file:
        json.dump(ancestor_map, json_file, indent=4, separators=(',', ': '))
    print("wrote out dict to file {}".format(file_molepro))

    print()

    # test the query string creator
    query_string = create_query_string('biolink:Gene', 'biolink:Gene', 'biolink:related_to')
    print("got query string {}".format(query_string))
    query_string = create_query_string('biolink:Gene', 'biolink:Gene')
    print("got query string {}".format(query_string))

    print()

    # test get all query strings
    print("testing getting predicate queries")
    with closing(requests.get(url_molepro_predicates)) as response_obj:
        response = response_obj.json()
        query_string_list = create_predicate_query_list(response)
        for item in query_string_list:
            print("query '{}'".format(item))


    print()

    # test query ancestry
    print("testing possible ancestry queries")
    query_list = build_query_descendant_list(ancestor_map, ["biolink:BiologicalEntity"], ["biolink:Gene"], ["biolink:correlated_with"])
    for query in query_list:
        print("got query '{}'".format(query))

    print()

    # test making arrays
    temp = None
    print("for {} got array {} with type {}".format(temp, make_into_array(temp), type(make_into_array(temp))))
    temp = "mustard"
    print("for {} got array {} with type {}".format(temp, make_into_array(temp), type(make_into_array(temp))))
    temp = ['ketchup']
    print("for {} got array {} with type {}".format(temp, make_into_array(temp), type(make_into_array(temp))))

    print()

    # test getting parts to query
    with open(file_query) as f:
        query_json = json.load(f)
        print("got query result {}".format(get_query_parts(query_json)))

    print()

    # test get all query strings
    print("testing query and ancestor overlap queries")
    with closing(requests.get(url_molepro_predicates)) as response_obj:
        predicate_json = response_obj.json()
        with open(file_chem_query) as f:
            query_json = json.load(f)
            overlap_list = get_all_overap_queries(ancestor_map, predicate_json, query_json)
            for item in overlap_list:
                print("got overlap query '{}'".format(item))
        print()
        with open(file_blank_query) as f:
            query_json = json.load(f)
            overlap_list = get_all_overap_queries(ancestor_map, predicate_json, query_json)
            for item in overlap_list:
                print("got overlap query '{}'".format(item))
        print()
        overlap_list = get_all_overap_queries_for_parts(ancestor_map, predicate_json, None, None, 'biolink:correlated_with')
        for item in overlap_list:
            print("got overlap query '{}'".format(item))



