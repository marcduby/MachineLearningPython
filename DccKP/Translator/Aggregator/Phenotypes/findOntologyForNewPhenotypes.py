
# imports 
import pandas as pd 
import requests 
import time
import os 
import sys
import logging
import pymysql as mdb


# dynamic lib
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
dir_code = "/home/javaprog/Code/PythonWorkspace/"
dir_data = "/home/javaprog/Data/Broad/"
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/Translator/TranslatorLibraries')
import translator_libs as tl

# constants 
file_rare_disease = '/home/javaprog/Data/Broad/Translator/RareDisease/DCC_GARD_RareDiseases.csv'
file_test_rare_disease = '/home/javaprog/Data/Broad/Translator/RareDisease/Test_DCC_GARD_RareDiseases.csv'
url_name_search = 'https://name-resolution-sri.renci.org/lookup?string={}'
DB_PASSWD = os.environ.get('DB_PASSWD')
DB_SCHEMA = 'tran_upkeep'
list_ontology = ['MONDO', 'EFO', 'UMLS', 'NCIT', 'HP']

# methods 
# def find_ontology(disease):
#     '''
#     will call REST api and will return ontology id if name exact match 
#     '''
#     # initialize
#     ontology_id = None

#     # call the url
#     response = requests.post(url_name_search.format(disease.replace("-", " ")))
#     output_json = response.json()

#     # loop through results, find first exact result
#     for key, values in output_json.items():
#         # print("key: {}".format(key))
#         # print("value: {}\n".format(values))
#         # do MONDO search first since easiest comparison
#         if 'MONDO' in key:
#             if disease.lower() in map(str.lower, values):
#                 ontology_id = key
#                 break

#     # return
#     return ontology_id

def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=DB_SCHEMA)

    # return
    return conn 

def get_new_phenotype_list(conn):
    '''
    get the list of new phenotypes in the aggregator but not yet in translator
    returns list of tuples (name, id)
    '''
    # initialize
    sql_select = """
    select id, phenotype_name, phenotype_id from tran_upkeep.agg_aggregator_phenotype 
    where in_translator = 'false' and ontology_id is null 
    and id in (255, 256)
    order by phenotype_name
    """

    # query the db
    cursor = conn.cursor()
    cursor.execute(sql_select)
    db_results = cursor.fetchall()

    # check
    if len(db_results) > 0:
        result = True

    # get the data
    if db_results:
        result = [(item[0], item[1], item[2]) for item in db_results]

    # return
    return result

def add_phenotype_ontology_id(conn, row_id, ontology_id):
    '''
    add in found ontology_id for the new disease
    '''
    # initialize
    sql_update = "update tran_upkeep.agg_aggregator_phenotype set ontology_id = %s where id = %s"

    # query the db
    cursor = conn.cursor()
    cursor.execute(sql_update, (ontology_id, row_id))


if __name__ == "__main__":
    # initialize
    list_phenotypes = []
    count = 0

    # get the connection
    db_connection = get_connection()

    # get the list of phenotypes that are not in the 
    list_phenotypes = get_new_phenotype_list(db_connection)
    print("got {} new phenotypes to add to translator".format(len(list_phenotypes)))

    # loop
    for (row_id, name, phenotype_id) in list_phenotypes:
        count = count + 1
        if count > 500:
            break
    
        # search for an ontology id
        ontology_id = tl.find_ontology(name, list_ontology)
        print("{} found for {} - '{}'".format(ontology_id, phenotype_id, name))

        # add in to table if not null
        if ontology_id:
            add_phenotype_ontology_id(db_connection, row_id, ontology_id)
            print("row {} - {} added for {} - {}".format(row_id, ontology_id, phenotype_id, name))

    # commit
    db_connection.commit()

# # get the phenotypes from 
# # read the file
# df_rare_disease = pd.read_csv(file_rare_disease, sep=',', header=0)
# print("after reading: \n{}".format(df_rare_disease.info()))

# # loop through rows and look for match for disease name 
# count = 0
# for index, row in df_rare_disease.iterrows():
#     ontology = row['ontology']
#     ontology_check = row['ontology_check']
#     if pd.isnull(ontology) and pd.isnull(ontology_check):
#         # log
#         print("no previous ontology for: {}".format(row['d.name']))
#         count += 1

#         # find ontology
#         result = find_ontology(row['d.name'])

#         # if found, log and set
#         if result is not None:
#             print("found ontology for: {} - {}\n".format(row['d.name'], result))
#             df_rare_disease.loc[df_rare_disease['d.name'] == row['d.name'], ['ontology']] = result
        
#         # log that checked
#         df_rare_disease.loc[df_rare_disease['d.name'] == row['d.name'], ['ontology_check']] = "yes"

#         # break if count reached
#         if count%10 == 0:
#             print("{} - data saved to file".format(count))
#             df_rare_disease.to_csv(file_rare_disease, sep=',', index=False)
#             # break

#         # sleep for throttling avoidance
#         # time.sleep(10)
    
# # log
# print("\nafter updating: \n{}".format(df_rare_disease.info()))

# # write out results 
# # df_rare_disease.to_csv(file_test_rare_disease, sep=',')
# df_rare_disease.to_csv(file_rare_disease, sep=',', index=False)
