
# imports
import requests

# constants
url_aggregator = "https://bioindex-dev.hugeamp.org/api/bio/query"


# set the string
phenotype = 'pparg'

def query_service(input_gene, url):
    ''' queries the service for disease/gene relationships '''
    # build the query
    query_string = """
    query {
        GeneAssociations(gene: "%s") {
            phenotype, gene, pValue
        }
    }
    """ % (input_gene)

    print(query_string)

    # query the service
    response = requests.post(url, data=query_string).json()

    # return
    return response

def get_phenotype_values(input_json, query_key):
    ''' will parse the graphql output and generate phenotype/pValue tupes list '''
    data = input_json.get('data').get(query_key)
    result = []

    # loop
    if data is not None:
        # result = [(item.get('phenotype'), item.get('pValue')) for item in data]
        result = [(item.get('phenotype'), item.get('pValue')) for item in data if item.get('pValue') <  0.000025]

    # rerurn
    return result

if __name__ == "__main__":
    # genes
    phenotypes = ['PPARG']

    for phenotype in phenotypes:
        resp = query_service(phenotype, url_aggregator)

        # print(f'\n{resp}')

    # get the phenotype data
    data = get_phenotype_values(resp, "GeneAssociations")
    print(f'got data size of {len(data)}')