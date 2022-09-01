
# imports
import pymysql as mdb
import os 
import json
import glob


# constants
dir_data = "/Users/mduby/Data/Broad/"
dir_data = "/home/javaprog/Data/Broad/"
dir_pathways = dir_data + "Translator/GeneticsPro/DataLoad/Pathways"
is_insert_data = True
is_update_data = True
DB_PASSWD = os.environ.get('DB_PASSWD')
db_pathway_table = "tran_upkeep.data_pathway_genes"
max_count = 2000

# db constants
schema_data_load = "tran_upkeep"
table_pathway_genes = "data_pathway_genes"
table_pathways = "data_pathway"

# sql statements
sql_insert = """insert into {} (pathway_id, gene_code) select id, %s from data_pathway where pathway_code = %s
    """.format(db_pathway_table)

sql_delete = "delete from {}".format(db_pathway_table)

# methods
def get_db_connection():
    '''
    opens a database connection
    '''
    # connect to the database
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=schema_data_load)

    # return
    return conn

def delete_pathway_genes(conn):
    '''
    will delete data from the pathway genes table
    '''
    sql_delete = "delete from {}.{}".format(schema_data_load, table_pathway_genes)

    # execute
    cursor = conn.cursor()
    cursor.execute(sql_delete)

    # commit
    conn.commit()

def delete_pathways(conn):
    '''
    will delete data from the pathways table
    '''
    sql_delete = "delete from {}.{}".format(schema_data_load, table_pathways)

    # execute
    cursor = conn.cursor()
    cursor.execute(sql_delete)
    
    # commit
    conn.commit()

def insert_pathway(conn, pathway_code, pathway):
    '''
    method to insert a pathway into the database
    '''
    sql_insert = """
    insert into {}.{} (pathway_code, pathway_name, exact_source, pmid, systematic_name, msig_url, gene_count)
    values(%s, %s, %s, %s, %s, %s, %s)
    """.format(schema_data_load, table_pathways)

    # create the update pathway name
    pathway['new_name'] = create_updated_name(pathway_code)

    # insert the pathway
    cursor = conn.cursor()
    cursor.execute(sql_insert, (pathway_code, pathway.get('new_name'), pathway.get('exactSource'), pathway.get('pmid'), pathway.get('systematicName'), pathway.get('msigdbURL'),
        len(pathway.get('geneSymbols'))))

    # insert the genes



def log_pathway_data_counts(conn):
    '''
    method to print out the number of pathways and pathway gene links in the database
    '''
    sql_count = "select count(id) from {}.{}"
    map_tables = {'pathways': table_pathways, 'pathway_genes': table_pathway_genes}

    # log the number
    cursor = conn.cursor()
    for key, value in map_tables.items():
        sql_to_run = sql_count.format(schema_data_load, value)
        cursor.execute(sql_to_run)
        results = cursor.fetchall()

        for row in results:
            print("for: {} got row count: {}".format(key, row[0]))
        # print("for: {} got row count: {}".format(key, results))

# functions
def create_updated_name(name, log=True):
    '''
    creates a human redable name from the given db name
    '''
    new_name = name.replace("_", " ")
    new_name = new_name.title()
    new_name = new_name.replace("Gomf ", "Gomf - ")
    new_name = new_name.replace("Gocc ", "Gocc - ")
    new_name = new_name.replace("Gobp ", "Gobp - ")
    new_name = new_name.replace("Reactome ", "Reactome - ")

    # log
    if log:
        print("for: {} got new name: {}".format(name, new_name))

    # return
    return new_name

# main
if __name__ == "__main__":
    # initialize
    list_pathways = []
    counter = 0

    # connect to the database
    conn = get_db_connection()

    # log the current db data
    log_pathway_data_counts(conn)

    # delete the existing rows in the db
    delete_pathway_genes(conn)
    delete_pathways(conn)
    print("deleted data\n")

    # get a list of the files
    dir_glob = "{}/*.json".format(dir_pathways)
    print("using pathway lookup: {}".format(dir_glob))
    list_file = [file for file in glob.glob(dir_glob)]
    # list_file = [file for file in os.listdir(dir_pathways) if file.endswith('.json')]

    # load the files
    for file_pathways in list_file:
        # with open(file_os, 'r') as file_pathways:
        # print(type(file_os))
        # file_pathways = dir_pathways + "/" + file_os

        # load the file
        print("loading pathway file: {}".format(file_pathways))
        with open(file_pathways) as file_json: 
            json_pathways = json.load(file_json)
            print("for file: {} \ngot: {} pathways\n".format(file_pathways, len(json_pathways)))


            # loop through pathways
            for key, row_pathway in json_pathways.items():
                # add counter
                counter = counter + 1
                if counter > max_count:
                    break

                # insert the pathway
                insert_pathway(conn, key, row_pathway)
                
                # log
                if counter % 500 == 0:
                    conn.commit()
                    print("inserted pathway: {} with code: {}".format(key, row_pathway['exactSource']))


        # log the current db data
        log_pathway_data_counts(conn)



        # add data
        # list_pathways = list_pathways + list_pathway_from_file


    # # insert all the new rows
    # counter = 0
    # for row in list_pathways:
    #     # get the list of genes
    #     list_genes = row['list_genes']

    #     for gene in list_genes:
    #         # insert
    #         cur.execute(sql_insert, (gene, row['id']))
    #         counter = counter + 1

    #         # commit every 100
    #         if counter % 100 == 0:
    #             print("{} - pathway gene added with code {} for pathway {}".format(counter, gene, row['id']))
    #             conn.commit()

    conn.commit()










# # methods
# def get_normalizer_data(curie_id, ontology, debug=True):
#     ''' calls the node normlizer and returns the name and asked for curie id '''
#     result_name, result_id = None, None
#     url = url_node_normalizer.format(curie_id)

#     # log
#     if debug:
#         print("looking up curie: {} - {}".format(curie_id, ontology))
#         print("looking up url: {}".format(url))

#     # call the normalizer
#     response = requests.get(url)
#     json_response = response.json()
#     if debug:
#         print(json_response)

#     # get the data from the response
#     try:
#         if json_response:
#             result_name = json_response.get(curie_id).get("id").get("label")
#             for item in json_response.get(curie_id).get("equivalent_identifiers"):
#                 if ontology in item.get("identifier"):
#                     result_id = item.get("identifier")
#                     break
#         else:
#             print("ERROR: got no response for curie {} and ontology {}".format(curie_id, ontology))
#     except:
#         print("ERROR: got no response for curie {} and ontology {}".format(curie_id, ontology))

#     # log
#     if debug:
#         print("got name: {}, curie id: {}".format(result_name, result_id))

#     # return
#     return result_name, result_id


# # load the data and display
# df_gencc = pd.read_csv(file_input, sep="\t")
# print("df head: \n{}".format(df_gencc.head(10)))
# print("df info: \n{}".format(df_gencc.info()))

# # subset the fields
# # df_filtered_gencc = df_gencc[['gene_curie', 'gene_symbol', 'disease_curie', 'disease_title', 'submitter_curie', 'submitter_title', 'submitted_as_assertion_criteria_url', 'moi_title']]
# df_filtered_gencc = df_gencc[['gene_curie', 'gene_symbol', 'disease_curie', 'disease_title', 'submitter_curie', 'submitter_title', 'moi_title', 'uuid', 'classification_title', 'submitted_as_pmids']]
# # df_filtered_gencc = df_gencc[['gene_curie', 'gene_symbol', 'disease_curie', 'disease_title', 'submitter_curie', 'submitter_title', 'moi_title']]
# print("df head: \n{}".format(df_filtered_gencc.head(50)))
# print("df info: \n{}".format(df_filtered_gencc.info()))

# # look at the unique sources
# a = df_filtered_gencc['submitter_title'].unique()
# print("unique sources: {}".format(sorted(a)))

# df_filtered_gencc = df_filtered_gencc.loc[~df_filtered_gencc['submitter_title'].isin(list_filter_out_source)]
# print("df head: \n{}".format(df_filtered_gencc.head(50)))
# print("df info: \n{}".format(df_filtered_gencc.info()))


# # create connection
# # conn = mdb.connect(host='localhost', user='root', password='this aint no password', charset='utf8', db='tran_genepro')
# conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db='tran_dataload')
# cur = conn.cursor()

# sql_insert = """insert into `data_gencc_gene_phenotype` (excel_id, gene, gene_hgnc_id, gene_annotation, phenotype, phenotype_mondo_id, 
#                 submitter, submitter_curie, gene_ncbi_id, phenotype_genepro_name)
#          values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
#     """

# sql_delete = "delete from data_gencc_gene_phenotype"

# sql_update = 'update data_gencc_gene_phenotype set score_classification = %s, publications = %s where excel_id = %s'

# if is_insert_data:
#     # delete all data
#     cur.execute(sql_delete)
#     print("deleted data\n")

#     # loop through rows
#     counter = 0
#     for index, row in df_filtered_gencc.iterrows():
#         # get the disease_name
#         disease_name, mondo_id = get_normalizer_data(row['disease_curie'], 'MONDO', debug=False)

#         # get the ncbi gene id
#         gene_name, gene_ncbi_id = get_normalizer_data(row['gene_curie'], 'NCBIGene', debug=False)

#         # if both disease and gene id
#         if disease_name and gene_ncbi_id:
#             # insert
#             cur.execute(sql_insert, (row['uuid'], row['gene_symbol'], row['gene_curie'], row['moi_title'], row['disease_title'], 
#                 row['disease_curie'], row['submitter_title'], row['submitter_curie'], gene_ncbi_id, disease_name))
#             counter = counter + 1

#             # commit every 10
#             if counter % 100 == 0:
#                 print("{} - gene {} with id {}".format(counter, row['gene_symbol'], disease_name))
#                 conn.commit()

#     conn.commit()

# # add in classification title
# if is_insert_data or is_update_data:
#     # update the 
#     # loop through rows
#     counter = 0
#     for index, row in df_filtered_gencc.iterrows():
#         publications = None
#         if not pd.isnull(row['submitted_as_pmids']):
#             publications = row['submitted_as_pmids']

#         # insert
#         cur.execute(sql_update, (row['classification_title'], publications, row['uuid']))
#         counter = counter + 1

#         # commit every 10
#         if counter % 100 == 0:
#             print("{} - gene {} with id {} and {}".format(counter, row['gene_symbol'], row['disease_curie'], row['classification_title']))
#             conn.commit()

#     conn.commit()
