

# imports
import requests
import json

# constants
URL_CHOD = "https://cohd-api.transltr.io/api/{}"
URI_TO_OMOP = "translator/biolink_to_omop"
URI_PREVALENCE = "frequencies/singleConceptFreq?dataset_id=1&q=192855"


# methods
def get_omop_for_list(list_curies, log=False):
    '''
    will query the cohd server for omop curies based on curies given
    '''
    # initialize
    map_results = {}
    url = URL_CHOD.format(URI_TO_OMOP)

    # call the service
    response = requests.post(url, json={'curies': list_curies})
    json_response = response.json()

    if log:
        print("ompo response: \n{}".format(json.dumps(json_response, indent=2)))
        # print("ompo response: {}".format(json_response))

    # loop over results
    for key, value in json_response.items():
        if value:
            map_results[key] = value.get('omop_concept_id')
        # else:
        #     map_results[key] = value

    # return
    return map_results


# main
if __name__ == "__main__":
    # data
    list_curies = [
        "HP:0002907",
        "HP:0012745",
        "HP:0005110"
    ]
    map_to_omop = {}

    # test the omop call
    map_to_omop = get_omop_for_list(list_curies=list_curies, log=False)
    print("got omop response: \n{}".format(json.dumps(map_to_omop, indent=2)))
