

# imports
import copy

# constants
MESSAGE = "message"
QUERY_GRAPH = "query_graph"
EDGES = "edges"
SUBJECT = "subject"
OBJECT = "object"
PREDICATES = "predicates"
NODES = "nodes"
IDS = "ids"
CATEGORIES = "categories"


# methods
def reverse_query(query, map_predicate, debug=False):
    ''' will check to see if query has id on object id only, then flips it if applicable; do nothing if not '''
    #initialize
    result = None
    subject = None
    object = None
    query_graph = None
    edge_id = None
    predicates = None

    # get the data
    if query.get(MESSAGE).get(QUERY_GRAPH):
        query_graph = query.get(MESSAGE).get(QUERY_GRAPH)
        if query_graph.get(EDGES):
            edges = query.get(MESSAGE).get(QUERY_GRAPH).get(EDGES)
            if len(edges) == 1:
                for key, value in edges:
                    edge_id = key
                    subject = edges.get(SUBJECT)
                    object = edges.get(OBJECT)
                    predicates = edges.get(PREDICATES)

                # test if id on object only
                if subject and object:
                    if query_graph.get(NODES).get(object).get(IDS) and not query_graph.get(NODES).get(subject).get(IDS):
                        # copy and flip
                        result = copy.deepcopy(query)

                        # flip
                        result.get(MESSAGE).get(QUERY_GRAPH).get(EDGES).get(edge_id).get(SUBJECT) = object
                        result.get(MESSAGE).get(QUERY_GRAPH).get(EDGES).get(edge_id).get(OBJECT) = subject
                        predicates_reverse = []
                        for item in predicates:
                            predicates_reverse.add(map_predicate.get(item))
                                    
    # return orginal and result
    return query, result


# main
if __name__ == "__main__":
    # load the input file

    # reverse

    # test
