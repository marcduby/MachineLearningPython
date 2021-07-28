

# imports
import json
import requests

# constants
URL_ONTOLOGY_KP = "https://stars-app.renci.org/sparql-kp/query"


# methods
def build_query(predicate, subject_category, subject_id, object_category, object_id):
    ''' will build a trapi v1.1 query '''
    edges = {"e00": {"predicates": [predicate], "subject": "n00", "object": "n01"}}
    nodes = {"n00": {"categories": [subject_category]}, "n01": {"categories": [object_category]}}
    if subject_id:
        nodes["n00"]["ids"] = [subject_id]
    if object_id:
        nodes["n01"]["ids"] = [object_id]
    message = {"query_graph": {"edges": edges, "nodes": nodes}}
    result = {"message": message}

    # return
    return result

def get_node_list(json_response):
    ''' will extract the nodes from the trapi v1.1 response'''
    result = []

    # get the nodes
    if json_response and json_response.get("message") and json_response.get("message").get("query_graph"):
        knowledge_graph = json_response.get("message").get("knowledge_graph")

        # loop
        if knowledge_graph.get("nodes"):
            for key, values in knowledge_graph.get("nodes").items():
                result.append(key)

    # return result
    return result

def query_service(url, query):
    ''' will do a post call to a service qith a trapi v1.1 query'''
    response = None

    # call
    response = requests.post(url, json=query).json()

    # return
    return response

def get_disease_descendants(disease_id, debug=False):
    ''' will query the trapi v1.1 ontology kp and return the descendant diseases '''
    # initialize
    list_diseases = [disease_id]
    json_query = build_query(predicate="biolink:subclass_of", subject_category="biolink:Disease", object_category="biolink:Disease", subject_id=None, object_id=disease_id)

    # print result
    if debug:
        print("the query is: \n{}".format(json.dumps(json_query, indent=2)))

    # query the KP and get the results
    response = query_service(URL_ONTOLOGY_KP, json_query)
    list_diseases = get_node_list(response)

    # log
    if debug:
        print("got the child disease list: {}".format(list_diseases))

    # return
    return list_diseases


# test
if __name__ == "__main__":
    disease_id = "MONDO:0007972"        # meniere's disease
    disease_id = "MONDO:0005267"        # heart disease
    get_disease_descendants(disease_id, debug=True)
    # json_query = build_query(predicate="biolink:subclass_of", subject_category="biolink:Disease", object_category="biolink:Disease", subject_id=None, object_id=)

    # # print result
    # print("the query is: \n{}".format(json.dumps(json_query, indent=2)))

    # # query the KP and get the results
    # response = query_service(URL_ONTOLOGY_KP, json_query)
    # list_diseases = get_node_list(response)
    # print("got the child disease list: {}".format(list_diseases))
