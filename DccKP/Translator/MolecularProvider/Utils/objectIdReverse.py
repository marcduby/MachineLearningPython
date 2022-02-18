

# imports
import copy
import json

# constants
dir_data = "/Users/mduby/Code/WorkspacePython/MachineLearningPython/DccKP/Translator/MolecularProvider/Utils/"
file_object_id = dir_data + "objectIdQuery.json"
file_subject_id = dir_data + "subjectIdQuery.json"
MESSAGE = "message"
KNOWLEDGE_GRAPH = "knowledge_graph"
QUERY_GRAPH = "query_graph"
EDGES = "edges"
SUBJECT = "subject"
OBJECT = "object"
PREDICATES = "predicates"
NODES = "nodes"
IDS = "ids"
CATEGORIES = "categories"
map_predicate = {"biolink:affected_by": "biolink:affects", 
    "biolink:condition_associated_with_gene": "biolink:gene_associated_with_condition",
    "biolink:gene_associated_with_condition": "biolink:condition_associated_with_gene"}

# methods
def reverse_response(response, map_predicate, debug=False):
    ''' will reverse the response data '''
    # TODO - flip the edge subject, object and predicates
    # TODO - replace the query graph with the original
    # initialize
    result = copy.deepcopy(response)
    subject = None
    object = None
    query_graph = None
    edge_id = None
    predicates = None

    # get the data
    if response.get(MESSAGE).get(KNOWLEDGE_GRAPH):
        knowledge_graph = response.get(MESSAGE).get(KNOWLEDGE_GRAPH)
        if knowledge_graph.get(EDGES):
            edges = knowledge_graph.get(EDGES)
            for key, value in edges.items():
                edge_id = key
                subject = value.get(SUBJECT)
                object = value.get(OBJECT)
                predicates = value.get(PREDICATES)

                # flip the response
                result[MESSAGE][KNOWLEDGE_GRAPH][EDGES][key][SUBJECT] = object
                result[MESSAGE][KNOWLEDGE_GRAPH][EDGES][key][OBJECT] = subject


def reverse_query(query, map_predicate, debug=False):
    ''' will check to see if query has id on object id only, then flips it if applicable; do nothing if not '''
    # initialize
    result = None
    subject = None
    object = None
    query_graph = None
    edge_id = None
    predicates = None
    is_flipped = False

    # get the data
    if query.get(MESSAGE).get(QUERY_GRAPH):
        query_graph = query.get(MESSAGE).get(QUERY_GRAPH)
        if query_graph.get(EDGES):
            edges = query_graph.get(EDGES)
            if len(edges) == 1:
                for key, value in edges.items():
                    edge_id = key
                    subject = value.get(SUBJECT)
                    object = value.get(OBJECT)
                    predicates = value.get(PREDICATES)

                if debug:
                    print("subject: {}, object: {} predicates: {}".format(subject, object, predicates))

                # test if id on object only
                if subject and object:
                    if query_graph.get(NODES).get(object).get(IDS) and not query_graph.get(NODES).get(subject).get(IDS):
                        # copy and flip
                        is_flipped = True
                        result = copy.deepcopy(query)

                        if debug:
                            print("result: \n{}".format(result))

                        # flip object/subject
                        result[MESSAGE][QUERY_GRAPH][EDGES][edge_id][SUBJECT] = object
                        result[MESSAGE][QUERY_GRAPH][EDGES][edge_id][OBJECT] = subject

                        # flip predicates
                        predicates_reverse = []
                        for item in predicates:
                            predicates_reverse.append(map_predicate.get(item))
                        result[MESSAGE][QUERY_GRAPH][EDGES][edge_id][PREDICATES] = predicates_reverse
                                    
    # return orginal and result
    return is_flipped, result


# main
if __name__ == "__main__":
    # load the input file
    with open(file_object_id) as f:
        json_object_id = json.load(f)

    # reverse
    is_flipped, json_reverse = reverse_query(json_object_id, map_predicate, debug=True)

    # test
    print("was flipped: {}, reverse: \n{}".format(is_flipped, json_reverse))

    # load the input file
    with open(file_subject_id) as f:
        json_object_id = json.load(f)

    # reverse
    is_flipped, json_reverse = reverse_query(json_object_id, map_predicate, debug=True)

    # test
    print("was flipped: {}, reverse: \n{}".format(is_flipped, json_reverse))    