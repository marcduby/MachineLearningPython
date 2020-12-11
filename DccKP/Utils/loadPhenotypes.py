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
load_pheno_df = phenotype_df.filter(['PH', 'Phenotype', 'Dichotomous', 'Category', 'Group', 'MONDO ID exact', 'EFO ID exact'])

# replace NANs
load_pheno_df = load_pheno_df.replace({np.nan: None})
print(load_pheno_df.head(25))

# save to database
conn = mdb.connect(host='localhost', user='root', password='this aint no password', charset='utf8', db='genetics_lookup')
cur = conn.cursor()

sql = """insert into `phenotype_lookup` (phenotype_code, phenotype, dichotomous, category, group_name, mondo_id, efo_id, tran_efo_id)
         values (%s, %s, %s, %s, %s, %s, %s, %s) 
    """

# loop through rows
counter = 0
for index, row in load_pheno_df.iterrows():
    tran_efo_id = None
    if row['EFO ID exact'] is not None:
        tran_efo_id = row['EFO ID exact'].replace("_", ":")
    cur.execute(sql, (row['PH'], row['Phenotype'], row['Dichotomous'], row['Category'], row['Group'], row['MONDO ID exact'], row['EFO ID exact'], tran_efo_id))
    counter = counter + 1

    # commit every 10
    if counter % 10 == 0:
        print("{} - phenotype {} with efo id {}".format(counter, row['PH'], row['EFO ID exact']))
        conn.commit()



