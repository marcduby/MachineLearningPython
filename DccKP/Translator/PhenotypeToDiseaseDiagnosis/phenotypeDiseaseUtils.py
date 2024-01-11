

# imports
import requests
import json
import pymysql as mdb
import os
import math

# constants
DB_PASSWD = os.environ.get('DB_PASSWD')
SCHEMA_PHE = "phe_disease"
URL_CHOD = "https://cohd-api.transltr.io/api/{}"
URI_TO_OMOP = "translator/biolink_to_omop"
URI_PREVALENCE = "frequencies/singleConceptFreq?dataset_id=1&q={}"
URI_PATIENT_COUNT = "metadata/patientCount?dataset_id=1"
SQL_SELECT_DISEASE = """
    select link.phenotype_curie, link.disease_curie, dis.name 
    from phen_phenotype_disease_link link, phen_disease dis 
    where link.disease_curie = dis.curie and link.phenotype_curie in ({})"""

# methods
def get_connection(schema=SCHEMA_PHE):
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=schema)

    # return
    return conn

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


def get_patient_count(log=False):
    '''
    will query the cohd server for the patient count
    '''
    # initialize
    result_count = 0
    url = URL_CHOD.format(URI_PATIENT_COUNT)

    # call the service
    response = requests.get(url)
    json_response = response.json()

    if log:
        print("count response: \n{}".format(json.dumps(json_response, indent=2)))
        # print("ompo response: {}".format(json_response))

    # loop over results
    if json_response.get('results'):
        if json_response.get('results').get('count'):
            result_count = json_response.get('results').get('count')

    # return
    return int(result_count)


def get_prevalence_for_list(list_curies, log=False):
    '''
    returns the prevalence for the given list of curies
    '''
    # initialize 
    map_results = {}

    # get omop curies
    map_phenotypes = get_omop_for_list(list_curies=list_curies, log=log)

    # flip the phenotype map
    map_temp = {}
    for key, value in map_phenotypes.items():
        map_temp[value] = key

    if log:
        print("got temp map: \n{}".format(json.dumps(map_temp, indent=2)))

    # call cohd service
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
        map_results[map_temp.get(omop_id)] = {'prevalence': item.get('concept_frequency'), 'omop_id': omop_id}

    # return
    return map_results


def get_disease_score_for_phenotype_list(conn, list_curies, log=False):
    '''
    will return a map of disease scores and their asscociated phenotypes/prevlance
    '''
    # initialize
    map_results = {}
    list_filtered_phenotypes = []

    # get the prevalance
    map_prevalence = get_prevalence_for_list(list_curies=list_curies, log=log)
    list_filtered_phenotypes = list(map_prevalence.keys())

    # get the list of diseases for the found phenotypes
    cursor = conn.cursor()
    placeholders = ", ".join(["%s"] * len(list_filtered_phenotypes))
    sql = SQL_SELECT_DISEASE.format(placeholders)
    if log:
        print("executing SQL: {}".format(sql))
    cursor.execute(sql, list_filtered_phenotypes)
    db_results = cursor.fetchall()
    for row in db_results:
        phenotype_curie = row[0]
        disease_curie = row[1]
        disease_name = row[2]
        if log:
            print("got row with phenotype: {} and disease: {} - {}".format(phenotype_curie, disease_curie, disease_name))
        # create map entry if none for disease
        if not map_results.get(disease_curie):
            map_results[disease_curie] = {'phenotypes': []}
        # build map
        map_results.get(disease_curie)['disease_name'] = disease_name
        map_results.get(disease_curie).get('phenotypes').append(phenotype_curie)
        

    # calculate score
    for key, value in map_results.items():
        score = 0
        for phenotype in value.get('phenotypes'):
            prevalence = map_prevalence.get(phenotype).get('prevalence')
            score = score - math.log(prevalence)
        map_results.get(key)['score'] = score

    # return
    return map_results


# main
if __name__ == "__main__":
    # data
    list_curies = [
        "HP:0002907",
        "HP:0012745",
        # "HP:0005110", 
        "HP:0000574",
        "HP:0002870",
        "HP:0034003"
    ]
    map_to_omop = {}

    # test the omop call
    map_to_omop = get_omop_for_list(list_curies=list_curies, log=False)
    print("got omop response: \n{}".format(json.dumps(map_to_omop, indent=2)))

    # test the prevalance call
    print()
    map_to_prevalence = get_prevalence_for_list(list_curies=list_curies, log=True)
    print("got prevalence response: \n{}".format(json.dumps(map_to_prevalence, indent=2)))

    # test retrieving the disease
    print()
    list_curies = ["HP:0000601","HP:0000403","HP:0000527","HP:0000400","HP:0000006","HP:0000369","HP:0000126","HP:0000143","HP:0000582","HP:0000463"]
    conn = get_connection()
    map_disease = get_disease_score_for_phenotype_list(conn=conn, list_curies=list_curies, log=False)
    print("got prevalence response: \n{}".format(json.dumps(map_disease, indent=2)))
