
# imports
import json
import requests 
from contextlib import closing

# constants
url_node_normalizer = "https://bl-lookup-sri.renci.org/bl/{}/ancestors?version={}"
url_molepro_predicates = 'https://translator.broadinstitute.org/molepro/trapi/v1.0/predicates'
file_molepro = '/home/javaprog/Data/Broad/Translator/Molepro/biolinkAncestry.json'
file_genepro = '/home/javaprog/Data/Broad/Translator/Genepro/biolinkAncestry.json'


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

if __name__ == "__main__":
    # get the gene ancestors
    gene_list = get_biolink_ancestors('Gene')
    for item in gene_list:
        print("got ancestor {}".format(item))

    print()

    # get the molepro predicate objects
    entity_list = get_entities_from_predicates(url_molepro_predicates)
    for item in entity_list:
        print("got entity: {}".format(item))

    print()

    # get the map of ancestors for molepro
    ancestor_map = build_ancestry_map(url_molepro_predicates)
    for key in list(ancestor_map.keys()):
        print("for ancestor {} got list {}".format(key, ancestor_map.get(key)))
    with open(file_molepro, 'w') as json_file:
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
    with closing(requests.get(url_molepro_predicates)) as response_obj:
        response = response_obj.json()
        query_string_list = create_predicate_query_list(response)
        for item in query_string_list:
            print("query {}".format(item))