
# imports
import pymysql as mdb
import requests 
import os
import json

# steps 
# 1 - delete old cache and upkeep cache status table 
# 1a - load curies into status table 
# 2 - query all curies
# 3 - for each 
# a - get synonyms (including current curie)
# b - get ancestors (do in batches of 30)
# 4 - save for each 30 batch and mark curies as done 

# constants 
DB_SCHEMA = "tran_test_202303"
DB_PASSWD = os.environ.get('DB_PASSWD')
DIR_DATA = "/home/javaprog/Data/Broad/Translator/PloverDbTest/GeneticsTestData"
FILE_NODES = "{}/geneticsNodes.jsonl".format(DIR_DATA)

# sql statements
SQL_SELECT_NODES = """
    select no.id, no.ontology_id, no.node_name, type.type_name
    from {}.comb_node_ontology no, {}.comb_lookup_type type
    where no.node_type_id = type.type_id
    and type.type_id in (1, 2)
"""


def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=DB_SCHEMA)

    return conn


def get_node_list(conn, log=False):
    ''' 
    will return all the KB nodes
    '''
    result = []
    sql_string = SQL_SELECT_NODES.format(DB_SCHEMA, DB_SCHEMA)

    # query the db
    cursor = conn.cursor()
    cursor.execute(sql_string)
    db_results = cursor.fetchall()

    # get the data
    if db_results:
        result = [{'id': item[1], 'name': item[2], 'category': item[3], 'all_names': [item[2]], 'all_categories': [item[3]], 'description': item[2], 'equivalent_curies': [item[1]], 'publications': []} for item in db_results]

    # return
    return result

def write_to_file(list_nodes, file_name, log=False):
    '''
    will write out the data to a jsonl file
    '''
    # Convert list of dictionaries to JSONL string
    jsonl_string = "\n".join(json.dumps(record) for record in list_nodes)

    # Optionally, write to a file
    with open(file_name, 'w') as file:
        file.write(jsonl_string)

    # log
    if log:
        print("wrote out to file: {}".format(file_name))


if __name__ == "__main__":
    # get the connection
    conn = get_connection()

    # get the nodes
    list_nodes = get_node_list(conn=conn)

    # write out the nodes
    write_to_file(list_nodes=list_nodes, file_name=FILE_NODES, log=True)

