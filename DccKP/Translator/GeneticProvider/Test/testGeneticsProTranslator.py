# imports
import requests 
import argparse

# method to call rest service
def call_translator(url, source_curie, source_type, target_type):
    query_map = {
    "message": {
        "query_graph": {
        "edges": [
            {
            "id": "e00",
            "source_id": "n00",
            "target_id": "n01",
            "type": "associated"
            }
        ],
        "nodes": [
            {
            "curie": source_curie,
            "id": "n00",
            "type": source_type
            },
            {
            "id": "n01",
            "type": target_type
            }
        ]
        }
    }
    }

    # get the REST response
    response = requests.post(query_url, json=query_map).json()

    # test the response
    print("for source: {}, source_type: {} and target_type: {}".format(source_curie, source_type, target_type))
    print("got number of response: {}\n".format(len(response.get('knowledge_graph').get('edges'))))

# base url
query_url = "http://localhost:7000/query"

# test the various entries
def test_service(url_to_test):
    # build the query lists
    # gene/disease
    source_curie = "EFO:0001360"
    source_type = "disease"
    target_type = "gene"
    call_translator(url_to_test, source_curie, source_type, target_type)

    source_curie = "NCBIGene:5468"
    source_type = "gene"
    target_type = "disease"
    call_translator(url_to_test, source_curie, source_type, target_type)

    # gene/phenotype
    source_curie = "EFO:0006336"
    source_type = "phenotypic_feature"
    target_type = "gene"
    call_translator(url_to_test, source_curie, source_type, target_type)

    source_curie = "NCBIGene:5468"
    source_type = "gene"
    target_type = "phenotypic_feature"
    call_translator(url_to_test, source_curie, source_type, target_type)

    # pathway/disease
    source_curie = "EFO:0001360"
    source_type = "disease"
    target_type = "pathway"
    call_translator(url_to_test, source_curie, source_type, target_type)

    source_curie = "GO:0014805"
    source_type = "pathway"
    target_type = "disease"
    call_translator(url_to_test, source_curie, source_type, target_type)

    # pathway/phenotype
    source_curie = "EFO:0008037"
    source_type = "phenotypic_feature"
    target_type = "pathway"
    call_translator(url_to_test, source_curie, source_type, target_type)

    source_curie = "GO:0061430"
    source_type = "pathway"
    target_type = "phenotypic_feature"
    call_translator(url_to_test, source_curie, source_type, target_type)


if (__name__ == "__main__"):
    # setup the argumant parser
    parser = argparse.ArgumentParser(description='Test a GeneticsPro Translator API.')
    parser.add_argument('-u', '--url', help='the url to test', default='http://localhost:7000/query', required=False)

    # get the args
    args = vars(parser.parse_args())

    # print the command line arguments
    print("arguments used: {}\n".format(args))

    # set the parameters
    if args['url'] is not None:
        query_url = args['url']

    # run the tests
    test_service(query_url)
