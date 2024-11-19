
# imports
import json
import requests
import logging
import sys

# constants
URL_CHOD = "https://cohd-api.transltr.io/api/{}"
URI_TO_OMOP = "translator/biolink_to_omop"
URI_PREVALENCE = "frequencies/singleConceptFreq?dataset_id=1&q={}"
URI_PATIENT_COUNT = "metadata/patientCount?dataset_id=1"

# logging
logging.basicConfig(level=logging.INFO, format=f'[%(asctime)s] - %(levelname)s - %(name)s %(threadName)s : %(message)s')
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)


# methods
def get_map_phenotype_prevalence(list_phenotypes, log=True):
    '''
    returns a map of the prevalence of the phenotypes given 
    '''
    # initialize
    map_prevalence = {}
    map_result = {}

    # get the prevalence
    # for row in list_phenotypes:
    #     map_prevalence[row] = 0.5
    map_prevalence = get_prevalence_for_list(list_curies=list_phenotypes, log=log)

    # log
    if log:
        logger.info("from COHD got prevalence map: {}".format(json.dumps(map_prevalence, indent=2)))

    # format into simple key/value map
    for key, value in map_prevalence.items():
        map_result[key] = value.get('prevalence')

    # return
    return map_result

def get_prevalence_for_list(list_curies, log=True):
    '''
    returns the prevalence for the given list of curies
    '''
    # initialize 
    map_results = {}

    # get omop curies
    map_phenotypes = get_omop_for_list(list_curies=list_curies, log=log)
    if log:
        logger.info("got OMOP mapping: {} for input phenotypes: {}".format(json.dumps(map_phenotypes, indent=2), list_curies))

    # flip the phenotype map
    map_temp = {}
    for key, value in map_phenotypes.items():
        map_temp[value.get('omop_id')] = key

    if log:
        logger.info("got OMOP to curie_id temp map: \n{}".format(json.dumps(map_temp, indent=2)))

    # call cohd service
    # make sure at least one phenotype has an OMOP result match
    if len(map_temp) > 0:
        str_input = ",".join(str(num) for num in map_temp.keys())
        url = URL_CHOD.format(URI_PREVALENCE.format(str_input))
        if log:
            print("Using prevalence URL: {}".format(url))
        response = requests.get(url)
        json_response = response.json()

        # loop
        json_results = json_response.get('results')
        for item in json_results:
            omop_id = item.get('concept_id')
            # omop_name = 
            map_results[map_temp.get(omop_id)] = {'prevalence': item.get('concept_frequency'), 'omop_id': omop_id, 'omop_name': map_phenotypes.get(map_temp.get(omop_id)).get('omop_name')}

    # return
    return map_results


def get_omop_for_list(list_curies, log=True):
    '''
    will query the cohd server for omop curies based on curies given
    '''
    # initialize
    map_results = {}
    url = URL_CHOD.format(URI_TO_OMOP)

    # log
    if log:
        logger.info("calling OMOP url: {} for curies: {}".format(url, list_curies))

    # call the service
    response = requests.post(url, json={'curies': list_curies})
    json_response = response.json()

    if log:
        logger.info("ompo response: \n{}".format(json.dumps(json_response, indent=2)))
        # print("ompo response: {}".format(json_response))

    # loop over results
    for key, value in json_response.items():
        if value:
            map_results[key] = {'omop_id': value.get('omop_concept_id'), 'omop_name': value.get('omop_concept_name')}
        # else:
        #     map_results[key] = value

    # log
    if log:
        logger.info("returning OMOP key map: {}".format(map_results))

    # return
    return map_results


if __name__ == "__main__":
    # seet phenotypes
    list_phenotypes = ["HP:0100785", "HP:0001250"]

    # get the prevalence
    result = get_map_phenotype_prevalence(list_phenotypes=list_phenotypes, log=True)
    print("\n\n\nMAIN -------------------------------\n")
    print("results: {}".format(json.dumps(result, indent=2)))
