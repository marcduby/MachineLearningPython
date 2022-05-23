

# imports
import argparse
import requests

# constants
url_name_resolver = "https://name-resolution-sri.renci.org/lookup?string={}"

# functions
def translate(list_input, ontology_prefix, sort_by_ontology=False, log=False):
    '''
    translate array of values using the translator name resolver
    will return multiple rows if multiple results returned for one name
    ex: 
        list_test_result = translate(list_test, 'NCBIGene', sort_by_ontology=True)
    get:
        [('MT-ND2', 'NCBIGene:56168'), ('MT-ND2', 'NCBIGene:387315')]
    '''
    # initialize
    list_result = []

    # query for the list of names
    for name in list_input:
        url_call = url_name_resolver.format(name)
        try:
            response = requests.post(url_call)
            output_json = response.json()
        except ValueError:
            print("got json error for {}, so skip".format(name))
            continue

        # parse
        for key, value in output_json.items():
            if ontology_prefix in key:
                list_result.append((name, key))

    if sort_by_ontology:
        list_result.sort(key = lambda x: int(x[1].split(":")[1]))

    # return
    return list_result

# main
if __name__ == "__main__":
    list_test = ['MT-ND1', 'MT-ND2']
    list_test_result = translate(list_test, 'NCBIGene', sort_by_ontology=True)

    for item in list_test_result:
        print("got result: {}".format(item))

