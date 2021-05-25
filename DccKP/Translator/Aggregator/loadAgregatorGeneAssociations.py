
# imports
import requests
import pymysql as mdb
from datetime import datetime

# constants
url_query_aggregator = "https://bioindex-dev.hugeamp.org/api/bio/query"
p_value_limit = 0.0025

def get_gene_list(conn):
    ''' will return all the gene codes that are in the translator DB '''
    result = []
    sql_string = "select node_code, ontology_id from comb_node_ontology where node_type_id = 2"

    # query the db
    cursor = conn.cursor()
    cursor.execute(sql_string)
    db_results = cursor.fetchall()

    # check
    if len(db_results) > 0:
        result = True

    # get the data
    if db_results:
        result = [item[0] for item in db_results]

    # return
    return result

def get_connection():
    ''' get the db connection '''
    conn = mdb.connect(host='localhost', user='root', password='yoyoma', charset='utf8', db='tran_test')

    # return
    return conn 

def load_phenotypes_reference(conn, phenotype_list):
    ''' add phenotypes to mysql phenotype table '''
    sql = """insert ignore into tran_dataload.agg_aggregator_phenotype (phenotype_id, phenotype_name, group_name)
            values (%s, %s, %s) 
        """
    cur = conn.cursor()

    i = 0
    # loop through rows
    for phenotype_id, phenotype_name, group_name in phenotype_list:
        i += 1
        if i % 20 == 0:
            print("disease {}".format(phenotype_id))

        cur.execute(sql,(phenotype_id, phenotype_name, group_name))
    conn.commit()

def load_phenotypes_to_translator(conn, phenotype_list):
    ''' add phenotypes to the translator  mysql phenotype table '''
    sql = """insert into comb_node_ontology (node_code, node_type_id, node_name)
            values (%s, 12, %s) 
        """
    cur = conn.cursor()

    i = 0
    # loop through rows
    for phenotype_id, phenotype_name, group_name in phenotype_list:
        i += 1
        if i % 20 == 0:
            print("disease {}".format(phenotype_id))

        # check if phenotype not loaded yet
        if not check_phenotype(conn, phenotype_id):
            cur.execute(sql,(phenotype_id, phenotype_name))
            print("loading phenotype/disease {} - {}".format(phenotype_id, phenotype_name))

    conn.commit()


def print_num_phenotypes_in_db(conn):
    ''' will query and print the count of phenotypes in the db '''
    sql = """
    select count(*) from comb_node_ontology where node_type_id in (1, 3, 12)
    """
    cursor = conn.cursor()
    count = 0

    # call the query
    cursor.execute(sql)
    db_results = cursor.fetchall()

    # get the data
    if db_results:
        count = db_results[0][0]

    # print
    print("the are {} phenotypes/diseases in the translator db".format(count))

def print_num_phenotypes_for_gene_in_db(conn, gene):
    ''' will query and print the count of phenotypes for the gene in the db '''
    sql = """
    select count(id) from comb_node_edge where source_code = %s and source_type_id = 2 and target_type_id in (1, 3, 12) and study_id = 1
    """
    cursor = conn.cursor()
    count = 0

    # call the query
    cursor.execute(sql, (gene))
    db_results = cursor.fetchall()

    # get the data
    if db_results:
        count = db_results[0][0]

    # print
    print("for {} the are {} phenotypes/diseases in the translator db".format(gene, count))

def query_gene_assocations_service(input_gene, url):
    ''' queries the service for disease/chem relationships '''
    # build the query
    query_string = """
    query {
        GeneAssociations(gene: "%s") {
            phenotype, gene, pValue
        }
    }
    """ % (input_gene)

    # query the service
    response = requests.post(url, data=query_string).json()

    # return
    return response

def get_phenotype_values(input_json, p_value_limit):
    ''' will parse the graphql output and generate phenotype/pValue tupes list '''
    query_key = 'GeneAssociations'
    data = input_json.get('data').get(query_key)
    result = []

    # loop
    if data is not None:
        # result = [(item.get('gene'), item.get('phenotype'), item.get('pValue')) for item in data]
        result = [(item.get('gene'), item.get('phenotype'), item.get('pValue')) for item in data if item.get('pValue') <  p_value_limit]

    # rerurn
    return result

def insert_or_update_gene_data(conn, association_list, gene, log=False):
    ''' will update or insert data for the gene/phenotype association '''
    edge_id = "magma_gene_" + datetime.today().strftime('%Y%m%d')
    sql_select = "select count(id) from comb_node_edge where source_code = %s and target_code = %s and source_type_id = 2 and target_type_id in (1, 3, 12)"
    sql_insert_front = """insert into comb_node_edge (edge_id, edge_type_id, source_code, source_type_id, target_code, target_type_id, score, score_type_id, study_id) 
        values(%s, 5, %s, 2, %s, (select node_type_id from comb_node_ontology where node_code = %s and node_type_id in (1, 3, 12)), %s, 8, 1)"""
    sql_insert_back = """insert into comb_node_edge (edge_id, edge_type_id, target_code, target_type_id, source_code, source_type_id, score, score_type_id, study_id)
        values(%s, 10, %s, 2, %s, (select node_type_id from comb_node_ontology where node_code = %s and node_type_id in (1, 3, 12)), %s, 8, 1)"""
    sql_update = """update comb_node_edge set score = %s 
        where (source_code = %s and target_code = %s and source_type_id = 2 and target_type_id in (1, 3, 12))
        or (target_code = %s and source_code = %s and target_type_id = 2 and source_type_id in (1, 3, 12))
    """
    sql_delete = """delete from comb_node_edge where study_id = 1 and
        ((source_code = %s and source_type_id = 2 and target_type_id in (1, 3, 12))
        or (target_code = %s and target_type_id = 2 and source_type_id in (1, 3, 12)))
    """
    cursor = conn.cursor()

    # TODO - first delete thwe rows for that gene if the date is more than a week old
    if True:
        # delete the rows
        cursor.execute(sql_delete, (gene, gene))

        # loop
        for gene, phenotype, p_value in association_list:
            if log:
                print("{} - {} - {}".format(gene, phenotype, p_value))
            # insert
            cursor.execute(sql_insert_front, (edge_id, gene, phenotype, phenotype, p_value))
            cursor.execute(sql_insert_back, (edge_id, gene, phenotype, phenotype, p_value))

        conn.commit()
        if log:
            print("committing gene {} data".format(gene))

def insert_all_gene_aggregator_data(conn, log=False):
    ''' will query the aggregator for all disease/phentype gene magma association data for all genes in the translator DB '''
    cursor = conn.cursor()
    sql_select = "select node_code from comb_node_ontology where node_type_id = 2 order by node_code"

    # get the list of genes
    cursor.execute(sql_select)
    db_results = cursor.fetchall()

    # loop for each gene
    for item in db_results:
        gene = item[0]

        # get the aggregator data
        result_json = query_gene_assocations_service(gene, url_query_aggregator)
        phenotype_list = get_phenotype_values(result_json, p_value_limit)

        # log
        if log:
            print("for {} got new gene associations of size {}".format(gene, len(phenotype_list)))
            print_num_phenotypes_for_gene_in_db(conn, gene)

        # insert into the db
        insert_or_update_gene_data(conn, phenotype_list, gene)

        # log
        if log:
            print_num_phenotypes_for_gene_in_db(conn, gene)
            
if __name__ == "__main__":
    # get the db connection
    conn = get_connection()

    # get the genes
    gene_list = get_gene_list(conn)

    # log
    print("got gene list of size {}".format(len(gene_list)))
    
    # test the check_phenotype method
    assert (len(gene_list) > 19000) == True

    # test gene list
    gene_list = ['PPARG']

    # log
    print_num_phenotypes_for_gene_in_db(conn, gene_list[0])

    # get the phenotypes pvalues for the gene from the bioindex
    result_json = query_gene_assocations_service(gene_list[0], url_query_aggregator)
    phenotype_list = get_phenotype_values(result_json, p_value_limit)
    print("for {} got new gene associations of size {}".format(gene_list[0], len(phenotype_list)))
    
    # # update the associations
    # insert_or_update_gene_data(conn, phenotype_list, gene_list[0])

    # # log
    # print_num_phenotypes_for_gene_in_db(conn, gene_list[0])

    # run the whole list
    insert_all_gene_aggregator_data(conn, log=True)

