
# imports
import requests
import pymysql as mdb

# constants
url_query_aggregator = "https://bioindex-dev.hugeamp.org/api/bio/query"

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
    select count(id) from comb_node_edge where source_code = %s and source_type_id = 2
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

def get_phenotype_values(input_json):
    ''' will parse the graphql output and generate phenotype/pValue tupes list '''
    query_key = 'GeneAssociations'
    data = input_json.get('data').get(query_key)
    result = []

    # loop
    if data is not None:
        result = [(item.get('gene'), item.get('phenotype'), item.get('pValue')) for item in data]
        # result = [(item.get('phenotype'), item.get('pValue')) for item in data if item.get('pValue') <  0.000025]

    # rerurn
    return result


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
    phenotype_list = get_phenotype_values(result_json)
    print("for {} got new gene associations of size {}".format(gene_list[0], len(phenotype_list)))
    


    