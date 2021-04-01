# imports
import requests

# constants
url_molepro = "https://translator.broadinstitute.org/molepro/trapi/v1.0/query"


class Link():
    def __init__(self, subject_id, object_id, subject_name=None, object_name=None):
        self.subject_id = subject_id
        self.object_id = object_id
        self.subject_name = subject_name
        self.object_name = object_name

    def __str__(self):
        return "subject: ({}, {}), object: ({},  {})\n".format(self.subject_id, self.subject_name, self.object_id, self.object_name)

    def __repr__(self):
        return self.__str__()

def parse_result(result):
    ''' parse the resulting json into a list of objects '''
    # result list
    list_links = []

    # get the nodes map
    map_nodes = result.get('message').get('knowledge_graph').get('nodes')
    # print(map_nodes)
    # print()

    # loop though edges
    list_edges = result.get('message').get('knowledge_graph').get('edges').values()
    list_links = [Link(edge.get('subject'), edge.get('object')) for edge in list_edges]

    # add node names
    for link in list_links:
        link.subject_name = map_nodes.get(link.subject_id).get('name')
        link.object_name = map_nodes.get(link.object_id).get('name')

    # return
    return list_links

def query_service(subject_id, url):
    ''' queries the service for disease/chem relationships '''
    # build the query
    query = {
        "message": {
            "query_graph": {
                "edges": {
                    "e00": {
                        "subject": "n00",
                        "object": "n01",
                        "predicate": "biolink:treated_by"
                    }
                },
                "nodes": {
                    "n00": {
                        "id": subject_id,
                        "category": "biolink:Disease"
                    },
                    "n01": {
                        "category": "biolink:ChemicalSubstance"
                    }
                }
            }
        }
    }

    # query the service
    response = requests.post(url, json=query).json()

    # return
    return response

def main():
    disease_id = "MONDO:0004975"

    # call the service
    result = query_service(disease_id, url_molepro)

    # parse the output
    list_links = parse_result(result)

    # print
    for link in list_links:
        print(link)


        
if __name__ == "__main__":
    main()