# imports
import requests 
import argparse

# method to call rest service
def call_translator(url, source_curie, source_type, target_type, edge_type):
    query_map = {
    "message": {
        "query_graph": {
        "edges": [
            {
            "id": "e00",
            "source_id": "n00",
            "target_id": "n01",
            "type": edge_type
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
    print("for source: {}, source_type: {}, target_type: {} and edge_type: {}".format(source_curie, source_type, target_type, edge_type))
    print("got number of response: {}\n".format(len(response.get('knowledge_graph').get('edges'))))

# base url
query_url = "https://translator.broadinstitute.org/molepro_reasoner/query"

# load predicates
predicates = {
    "chemical_substance": {
        "assay": [
            "has_evidence"
        ],
        "chemical_substance": [
            "correlated_with",
            "has_metabolite"
        ],
        "disease": [
            "treats"
        ],
        "gene": [
            "affects",
            "correlated_with"
        ],
        "molecular_entity": [
            "affects"
        ]
    },
    "disease": {
        "chemical_substance": [
            "treated_by"
        ]
    },
    "gene": {
        "chemical_substance": [
            "affected_by",
            "correlated_with"
        ],
        "gene": [
            "related_to",
            "correlated_with"
        ]
    }
}

curie_map = {'disease': ['MONDO:0007455'],
                'gene': ['NCBIGene:1803', 'HGNC:4556'],
                'chemical_substance': ['ChEMBL:CHEMBL25']}

# loop through the predicates and test
for source_type in predicates.keys():
    for target_type in predicates.get(source_type).keys():
        for edge_type in predicates.get(source_type).get(target_type):
            if source_type in curie_map:
                for source_curie in curie_map.get(source_type):
                    call_translator(query_url, source_curie, source_type, target_type, edge_type)

