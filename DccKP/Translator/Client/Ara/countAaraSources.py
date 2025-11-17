

# imports
import requests
import json


# constants
URL_ARAGORN = "https://aragorn.transltr.io/aragorn/query"
URL_ARAX = "https://arax.transltr.io/api/arax/v1.4/query"
MAP_ARAS = {"aragorn": URL_ARAGORN, 'arax': URL_ARAX}

MAP_QUERY_TRAPI =   {
    "workflow": [
        {
            "id": "lookup"
        }
    ],
    "message": {
      "query_graph": {
        "edges": {
          "e00": {
            "subject": "source",
            "object": "target"
          }
        },
        "nodes": {
          "target": {
            "categories": ["biolink:Gene"]
          },
          "source": {
            "ids": ["MONDO:0011936"],
            "categories": ["biolink:DiseaseOrPhenotypicFeature"],
            "set_interpretation": "BATCH"
          }
        }
      }
    }
  }


# methods
# def get_trapi_payload()
def calculate_ara_sources(map_json, log=False):
    '''
    gets the count of soruyces for the given trapi result
    '''
    # initialize
    map_counts = {}

    # get the message
    map_kg = map_json.get('message', {}).get('knowledge_graph', {})

    # Iterate over all edges and count resource_ids in sources
    for edge_id, edge_data in map_kg["edges"].items():
        sources = edge_data.get("sources", [])
        for source in sources:
            if source.get('resource_role', '') == 'aggregator_knowledge_source':
                resource_id = source.get("resource_id")
                if resource_id:
                    map_counts[resource_id] = map_counts.get(resource_id, 0) + 1


    # Print the results
    if log:
        print("Resource ID -> Count")
        for resource_id, count in map_counts.items():
            print(f"{resource_id} -> {count}")


    # return
    return map_counts


def query_trapi_ara(url_ara, json_payload, log=True):
    '''
    makes a REST call to the trapi url and returns the result
    '''
    # query the service
    response = requests.post(url_ara, json=json_payload).json()

    # return
    return response


# main
if __name__ == "__main__":
    for key, value in MAP_ARAS.items():
        # query the ara
        json_result = query_trapi_ara(url_ara=value, json_payload=MAP_QUERY_TRAPI)

        # get the counts
        map_counts = calculate_ara_sources(map_json=json_result, log=False)

        # print
        print("for ara: {}, got counts: \n{}".format(key, json.dumps(map_counts, indent=2)))

