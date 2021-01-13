# 

# imports
import pandas as pd 
import pymysql as mdb
import numpy as np 

# file location
file_location = "/home/javaprog/Data/Broad/Shared/phenoOntology.tsv"

# load the file and inspect
phenotype_df = pd.read_csv (file_location, sep = '\t')
print(phenotype_df.head(25))

# filter to columns needed
load_pheno_df = phenotype_df.filter(['PH', 'Phenotype', 'Dichotomous', 'Category', 'Group', 'MONDO ID exact', 'MONDO term exact', 'EFO ID exact', 'EFO term exact'])

# replace NANs
load_pheno_df = load_pheno_df.replace({np.nan: None})
print(load_pheno_df.head(25))

# save to database
# conn = mdb.connect(host='localhost', user='root', password='this aint no password', charset='utf8', db='genetics_lookup')
conn = mdb.connect(host='localhost', user='root', password='yoyoma', charset='utf8', db='tran_genepro')
cur = conn.cursor()

sql = """insert into `phenotype_lookup` (phenotype_code, phenotype, dichotomous, 
        category, group_name, mondo_id, tran_mondo_id, mondo_name, efo_id, tran_efo_id, efo_name,
        tran_lookup_id, tran_lookup_name)
         values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
    """

# loop through rows
counter = 0
for index, row in load_pheno_df.iterrows():
    tran_efo_id = None
    tran_mondo_id = None
    tran_lookup_id = None
    tran_lookup_name = None
    if row['EFO ID exact'] is not None:
        tran_efo_id = row['EFO ID exact'].replace("_", ":")
        tran_lookup_id = tran_efo_id
        tran_lookup_name = row['EFO term exact']
    if row['MONDO ID exact'] is not None:
        tran_mondo_id = row['MONDO ID exact'].replace("_", ":")
        if tran_lookup_id is None:
            tran_lookup_id = tran_mondo_id
            tran_lookup_name = row['MONDO term exact']

    # run the sql
    cur.execute(sql, (row['PH'], row['Phenotype'], row['Dichotomous'], row['Category'], row['Group'], 
        row['MONDO ID exact'], tran_mondo_id, row['MONDO term exact'], row['EFO ID exact'], tran_efo_id, row['EFO term exact'], 
        tran_lookup_id, tran_lookup_name))
    counter = counter + 1

    # commit every 10
    if counter % 10 == 0:
        print("{} - phenotype {} with efo id {}".format(counter, row['PH'], row['EFO ID exact']))
        conn.commit()



