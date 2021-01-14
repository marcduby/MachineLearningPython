# 

# imports
import pandas as pd 
import pymysql as mdb
import numpy as np 

# file location
file_location = "/home/javaprog/Data/Broad/Shared/phenoOntology.tsv"

# method to build map
def get_ontology_map(df, ontology_key, name_key, input_id_map, input_code_map, add_duplicate_phenotypes=False):
    """will take in a df and populate the map based on the key given"""

    # loop
    for index, row in df.iterrows():
        key = row[ontology_key]
        code = row['PH']

        # if key not yet in map
        if key not in input_id_map:
            if add_duplicate_phenotypes or code not in input_code_map:
                temp_map = {'id': key, 'code': code, 'type': ontology_key, 'name': row[name_key], 'dichotomous': row['Dichotomous'], 'category': row['Category'], 'group': row['Group']}
                input_id_map[key] = temp_map
                input_code_map[code] = temp_map

    # return
    return input_id_map, input_code_map

# load the file and inspect
phenotype_df = pd.read_csv (file_location, sep = '\t')
print(phenotype_df.head(25))

# replace NANs
load_pheno_df = phenotype_df.replace({np.nan: None})
# print(load_pheno_df.head(25))


# build map with only unique ontology IDs
ontology_map = {}
code_map = {}

# load efo exact
ontology_map, code_map = get_ontology_map(load_pheno_df, 'EFO ID exact', 'EFO term exact', ontology_map, code_map)
print("got efo exact ontology length of: {} and code length of: {}".format(len(ontology_map), len(code_map)))

# load mondo exact
# ontology_map, code_map = get_ontology_map(load_pheno_df, 'MONDO ID exact', 'MONDO term exact', ontology_map, code_map, True)
ontology_map, code_map = get_ontology_map(load_pheno_df, 'MONDO ID exact', 'MONDO term exact', ontology_map, code_map)
print("got mondo exact ontology length of: {} and code length of: {}".format(len(ontology_map), len(code_map)))

# load efo closest
ontology_map, code_map = get_ontology_map(load_pheno_df, 'EFO ID closest parent', 'EFO term closest parent', ontology_map, code_map)
print("got efo closest ontology length of: {} and code length of: {}".format(len(ontology_map), len(code_map)))

# load mondo exact
ontology_map, code_map = get_ontology_map(load_pheno_df, 'MONDO ID closest parent', 'MONDO term closest parent', ontology_map, code_map)
print("got mondo closest ontology length of: {} and code length of: {}".format(len(ontology_map), len(code_map)))


# save to database
# conn = mdb.connect(host='localhost', user='root', password='this aint no password', charset='utf8', db='genetics_lookup')
cur = conn.cursor()

sql = """insert into `phenotype_id_lookup` (phenotype_code, ontology_name, dichotomous, 
        category, group_name,
        tran_lookup_id, tran_lookup_name)
         values (%s, %s, %s, %s, %s, %s, %s) 
    """

# loop through rows
counter = 0
for item in ontology_map.values():
    tran_lookup_id = None
    if item['id'] is not None:
        tran_lookup_id = item['id'].replace("_", ":")

    # run the sql
    cur.execute(sql, (item['code'], item['type'], item['dichotomous'], item['category'], item['group'], tran_lookup_id, item['name']))
    counter = counter + 1

    # commit every 10
    if counter % 10 == 0:
        print("{} - phenotype {} with efo id {}".format(counter, item['code'], tran_lookup_id))
        conn.commit()



