
# STEP 01 - LOAD THE DATA INTO A DATA TABLE, CALCULATE PROBABILITY AS WELL

# imports
import pandas as pd 
import pymysql as mdb
import requests 
import numpy as np
import math 

# constants
file_input = "/home/javaprog/Data/Broad/Translator/Genebass/Filtered/part-00000-2ff22837-6b57-4167-b3f8-f870919ba8cb-c000.csv"
list_filter_out_source = ['ClinGen']
url_node_normalizer = "https://nodenormalization-sri.renci.org/1.1/get_normalized_nodes?curie={}"
is_insert_data = True
is_update_data = True

# methods
# methods
def calculate_abf(standard_error, effect_size, variance=0.396):
    ''' calculates the approximate bayes factor '''
    V = standard_error ** 2

    # calculate result
    left_side = math.sqrt(V / (V + variance))
    right_side = math.exp((variance * effect_size ** 2) / (2 * V * (V + variance)))
    result = left_side * right_side

    # return
    return result

def convert_abf_to_probability(abf):
    ''' converts the approximate bayes factor to a probability '''
    PO = (0.05 / 0.95) * abf
    probability = PO / (1 + PO)

    # return
    return probability

def get_normalizer_data(curie_id, ontology, debug=True):
    ''' calls the node normlizer and returns the name and asked for curie id '''
    result_name, result_id = None, None
    url = url_node_normalizer.format(curie_id)

    # log
    if debug:
        print("looking up curie: {} - {}".format(curie_id, ontology))
        print("looking up url: {}".format(url))

    # call the normalizer
    response = requests.get(url)
    json_response = response.json()
    if debug:
        print(json_response)

    # get the data from the response
    try:
        if json_response:
            result_name = json_response.get(curie_id).get("id").get("label")
            for item in json_response.get(curie_id).get("equivalent_identifiers"):
                if ontology in item.get("identifier"):
                    result_id = item.get("identifier")
                    break
        else:
            print("ERROR: got no response for curie {} and ontology {}".format(curie_id, ontology))
    except:
        print("ERROR: got no response for curie {} and ontology {}".format(curie_id, ontology))

    # log
    if debug:
        print("got name: {}, curie id: {}".format(result_name, result_id))

    # return
    return result_name, result_id

# main program
if __name__ == "__main__":
    # load the data and display
    df_genebass = pd.read_csv(file_input, sep="\t")
    print("df head: \n{}".format(df_genebass.head(10)))
    print("df info: \n{}".format(df_genebass.info()))

    # drop the na rows
    print("df shape before dropping NA rows: {}".format(df_genebass.shape))
    df_genebass.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_genebass.dropna(subset = ['p_value', 'se', 'beta'], inplace=True)
    print("df shape after dropping NA rows: {}".format(df_genebass.shape))

    # create connection
    # conn = mdb.connect(host='localhost', user='root', password='this aint no password', charset='utf8', db='tran_genepro')
    conn = mdb.connect(host='localhost', user='root', password='yoyoma', charset='utf8', db='tran_dataload')
    cur = conn.cursor()

    sql_insert = """insert into `data_genebass_gene_phenotype` (gene, phenotype_genebass, phenotype_ontology_id, pheno_num_genebass, pheno_coding_genebass,
                    pvalue, standard_error, beta, abf, probability)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
        """

    sql_delete = "delete from data_genebass_gene_phenotype"

    # sql_update = 'update data_gencc_gene_phenotype set score_classification = %s, publications = %s where excel_id = %s'

    if is_insert_data:
        # delete all data
        cur.execute(sql_delete)
        print("deleted data\n")

        # loop through rows
        counter = 0
        for index, row in df_genebass.iterrows():
            # if have numbers
            if row['p_value'] and row['se'] and row['beta']:
                # insert
                # cur.execute(sql_insert, (row['gene'], row['pheno'], row['pheno_id'], row['p_code'], row['p_coding'], 
                #     row['p_value'], row['se'], row['beta']))
                # counter = counter + 1

                # calculate abf
                abf = calculate_abf(row['se'], row['beta'])
                probability = convert_abf_to_probability(abf)

                # get the coding
                pheno_coding = None
                if not pd.isna(row['p_coding']):
                    pheno_coding = row['p_coding']

                try:
                    cur.execute(sql_insert, (row['gene'], row['pheno'], row['pheno_id'], row['p_code'], pheno_coding, 
                        row['p_value'], row['se'], row['beta'], abf, probability))
                    counter = counter + 1
                except (mdb.err.ProgrammingError, OverflowError):
                    print("{} - got mysql error for gene {} with id {}, pvalue: {}, se: {}, beta: {}, code: {}, coding: {}, abf: {}, prob: {}".format(counter, row['gene'], row['pheno_id'], 
                        row['p_value'], row['se'], row['beta'], row['p_code'], row['p_coding'], abf, probability))
                    break

                # commit every 10
                if counter % 10000 == 0:
                    print("{} - gene {} with id {}, pvalue: {}, se: {}, beta: {}, code: {}, coding: {}, prob: {}".format(counter, row['gene'], row['pheno_id'], 
                        row['p_value'], row['se'], row['beta'], row['p_code'], row['p_coding'], probability))
                    conn.commit()


        conn.commit()

