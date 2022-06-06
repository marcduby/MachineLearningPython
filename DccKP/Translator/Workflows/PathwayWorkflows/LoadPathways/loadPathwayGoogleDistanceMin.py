
# imports
import pymysql as mdb
import os 
import json


# constants
dir_data = "/Users/mduby/Data/Broad/"
dir_data = "/home/javaprog/Data/Broad/"
file_pathways = dir_data + "Translator/Workflows/MiscQueries/ReactomeLipidsDifferentiation/GoogleDistancePathways/pathwayGoogleDistanceMin.json"
is_insert_data = True
is_update_data = True
DB_PASSWD = os.environ.get('DB_PASSWD')
db_pathway_table = "tran_upkeep.data_pathway_similarity"
max_count = 50000000000000

# sql statements
sql_update = """update {} set google_distance_min = %s where subject_pathway_code = %s and object_pathway_code = %s
    """.format(db_pathway_table)

# sql_delete = "delete from {}".format(db_pathway_table)


# main
if __name__ == "__main__":
    # load the file
    with open(file_pathways) as file_json: 
        json_pathways = json.load(file_json)
    list_pathways = json_pathways.get('results')
    print("got {} pathways google distance".format(len(list_pathways)))

    # connect to the database
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db='tran_upkeep')
    cur = conn.cursor()

    # insert all the new rows
    counter = 0
    for row in list_pathways:
        # insert
        # print(sql_update)
        # print("{} - {} - {}".format(row['google_distance_min'], row['subject_id'], row['object_id']))
        cur.execute(sql_update, (row['google_distance_min'], row['subject_id'], row['object_id']))
        cur.execute(sql_update, (row['google_distance_min'], row['object_id'], row['subject_id']))
        counter = counter + 1

        # commit every 10
        if counter % 1000 == 0:
            print("{} - pathway updated with id {} and {}".format(counter, row['subject_id'], row['object_id']))
            conn.commit()

        # break if trigger reached
        if counter > max_count:
            break

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
