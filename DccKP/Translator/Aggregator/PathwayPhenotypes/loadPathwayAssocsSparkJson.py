
# imports
import glob
import json
import pymysql as mdb
from datetime import datetime
import os

# constants
dir_pathway_associations = "/home/javaprog/Data/Broad/dig-analysis-data/out/magma/pathway-associations/CK/part-00000-3d610d35-dcad-442a-94d7-c6c7224a2b87-c000.json"
dir_pathway_associations = "/home/javaprog/Data/Broad/dig-analysis-data/out/magma/pathway-associations/*/part-*.json"
DB_PASSWD = os.environ.get('DB_PASSWD')
DB_SCHEMA = "tran_upkeep"

# methods
def main():
    # get the database connection
    conn = get_connection()

    # delete the existing data
    delete_pathway_associations(conn)

    # read the file list
    list_files = [file for file in glob.glob(dir_pathway_associations)]
    print("got list of files of size: {}".format(len(list_files)))

    # loop through files
    for file_name in list_files:
        print("file: {}".format(file_name))

        # load the json line by line
        with open(file_name, 'r') as file_pathway:
            list_pathways = []
            for line in file_pathway:
                # print(line.rstrip())
                dict_pathway = json.loads(line.rstrip())
                # print(dict_pathway)
                list_pathways.append(dict_pathway)

            # load the data
            load_pathway_associations(conn, list_pathways)

def load_pathway_associations(conn, list_pathway_assoc):
    ''' 
    add pathway/phenotype associations from the agregator results
    '''
    sql_insert = """
        insert ignore into {}.agg_pathway_phenotype (pathway_code, phenotype_code, number_genes, beta, beta_standard_error, standard_error, p_value)
            values (%s, %s, %s, %s, %s, %s, %s) 
        """.format(DB_SCHEMA)

    cur = conn.cursor()

    i = 0
    # loop through rows
    for pathway_association in list_pathway_assoc:
        phenotype = pathway_association.get('phenotype')
        pathway = pathway_association.get('pathwayName')
        pValue = pathway_association.get('pValue')

        # log
        i += 1
        if i % 2000 == 0:
            print("pathway: {}, phenotype: {}, pValue: {}".format(pathway, phenotype, pValue))

        cur.execute(sql_insert, (pathway, phenotype, pathway_association.get('numGenes'), pathway_association.get('beta'),
            pathway_association.get('betaStdErr'), pathway_association.get('std_err'), pValue))

    # commit
    conn.commit()

def delete_pathway_associations(conn):
    ''' 
    delete the pathway/phenotype associations from the agregator load table
    '''
    sql_delete = """delete from {}.agg_pathway_phenotype 
        """.format(DB_SCHEMA)

    # delete the data
    cur = conn.cursor()
    cur.execute(sql_delete)

    # commit
    conn.commit()

def get_connection():
    ''' get the db connection '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=DB_SCHEMA)

    # return
    return conn 



# main
if __name__ == "__main__":
    main()
