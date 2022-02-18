
# STEP 02 - LOAD THE PHENOTYPES FROM AN EXCEL SPREADSHEET AND FIND THE CURIES WE CAN THROUGH THE NODE NORMALIZER

# imports
import pandas as pd
import numpy as np
import requests 
import logging
import sys 
import time
import re

# constants
url_node_normalizer = 'https://name-resolution-sri.renci.org/lookup?string={}'
file_phenotypes = '/home/javaprog/Data/Broad/Translator/Genebass/ukBiobankCuries.tsv'
number_curies = 10
sleep_time = 1
list_prefix = ['MONDO', 'EFO', 'NCIT', 'UMLS', 'HP', 'MESH']
leading_code = re.compile('^[A-Z][\d*\s]')
regex_has_letter = re.compile('[\D]')
list_word_ignore = ['coffee', 'tea', 'milk', 'cereal', 'butter', 'country', 'population', 'ethnic', 'workplace', 'worked', 'shifts', 'pct', 'consultant']
list_word_include = ['operative', 'operation', 'depression', 'cancer', 'non-cancer']

# logger
logging.basicConfig(level=logging.INFO, format=f'[%(asctime)s] - %(levelname)s - %(name)s: %(message)s')
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)

# methods
def get_normalizer_data(name, list_ontology, debug=True):
    ''' calls the node normalizer and returns the name and asked for curie id '''
    result_id = None
    url = url_node_normalizer.format(name)

    # log
    if debug:
        logger.info("looking up name: '{}' - {}".format(name, list_ontology))
        logger.info("looking up url: {}".format(url))

    # call the normalizer
    response = requests.post(url)
    json_response = response.json()
    # if debug:
    #     logger.info(json_response)

    # get the data from the response
    try:
        if json_response:
            for ontology in list_ontology:
                for key, value in json_response.items():
                    if ontology in key:
                        for item in value:
                            if  item.lower() == name.lower():
                                result_id = key
                                break
                if result_id:
                    break
        else:
            logger.info("got empty response for name '{}' and ontology {}".format(name, list_ontology))
    except:
        logger.error("got no/exception response for name '{}' and ontology {}".format(name, list_ontology))

    # log
    if debug:
        if result_id:
            logger.info("for name: '{}', got curie id: {}".format(name, result_id))
        else:
            logger.info("got None curie for name: '{}', got: NONE".format(name))

    # return
    return result_id

# main
if __name__ == "__main__":
    # get cache map
    map_cache = {}

    # load the data and display
    df_curie = pd.read_csv(file_phenotypes, sep="\t")
    # print("df head: \n{}".format(df_curie.head(10)))
    print("df info: \n{}".format(df_curie.info()))

    # filter out the columns
    # df_curie = df_curie[['trait_type', 'phenocode', 'coding', 'description', 'curie_id']]
    print("df head: \n{}".format(df_curie.head(10)))
    print("df info: \n{}".format(df_curie.info()))

    # add curie
    df_curie['coding_integer'] = df_curie['coding']
    df_curie[['coding_integer']] = df_curie[['coding_integer']].fillna(value=0)
    if 'curie_id' not in df_curie:
        # assume all three not in df
        df_curie['curie_id'] = u''
        df_curie['curie_description_used'] = u''
        df_curie['curie_provenance'] = u''

    # loop through rows, find curies
    count = 0
    for index, row in df_curie.iterrows():
        if (pd.isnull(row['curie_id']) or row['curie_id'] == u'') \
                and not pd.isnull(row['description']) \
                and not any(word in row['description'].lower() for word in list_word_ignore):
            # get row data
            row_description = row['description']
            row_coding = row['coding_integer']
            row_coding_description = row['coding_description']

            # initialize vars
            curie_id_description = None
            curie_id_coding = None
            curie_provenance = None
            # if count > number_curies:
            #     break

            # if have coding number look using coding description
            # if row_coding and (regex_has_letter.matches(row_coding) or int(row_coding) > 0) and row_coding_description and row_coding_description != 'NA' and not pd.isnull(row_coding_description):
            if row_coding and row_coding_description and row_coding_description != 'NA' and not pd.isnull(row_coding_description):
                if row_description and any(word in row_description.lower() for word in list_word_include):
                    # get the phenotype curie
                    time.sleep(sleep_time)
                    curie_id_coding = get_normalizer_data(row_coding_description, list_ontology=list_prefix)
                    count += 1

            if not curie_id_coding:
                # take out leading codes from some phenotypes
                if leading_code.match(row_description):
                    logger.info("got leading code for: {}".format(row_description))
                    row_description = row_description[4:]
                    logger.info("convert string to: {}".format(row_description))

                # look into the cache; if not there, call rest service
                if map_cache.get(row_description):
                    curie_id_description = map_cache.get(row_description)
                else:                    
                    # get the phenotype curie
                    time.sleep(sleep_time)
                    curie_id_description = get_normalizer_data(row_description, list_ontology=list_prefix)
                    if not curie_id_description:
                        curie_id_description = 'NA'
                    map_cache[row_description] = curie_id_description
                    count += 1

            # # assign if not null
            if curie_id_coding:
                df_curie.at[index, 'curie_id'] = curie_id_coding
                df_curie.at[index, 'curie_description_used'] = row_coding_description
                df_curie.at[index, 'curie_provenance'] = 'coding_description'
                logger.info("{} - used description coding".format(count))
            else:
                df_curie.at[index, 'curie_id'] = curie_id_description
                df_curie.at[index, 'curie_description_used'] = row_description
                df_curie.at[index, 'curie_provenance'] = 'main_description'
                logger.info("{} - used description regular".format(count))

            # add space for better monitoring
            print()

        if count % 30 == 0:    
            # write out file
            logger.info("writing out file: {}".format(file_phenotypes))
            df_curie.to_csv(file_phenotypes, sep="\t", index=False)

    # one last data point
    print("df info: \n{}".format(df_curie.info()))
