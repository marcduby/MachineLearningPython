
# imports
import pandas as pd 
import pymysql as mdb
import requests 
import numpy as np
import os 
import time
import logging
import sys

# logging
logging.basicConfig(level=logging.INFO, format=f'[%(asctime)s] - %(levelname)s - %(name)s : %(message)s')
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)

# constants
list_genes = ['NCBIGene:1557', 'NCBIGene:1544', 'NCBIGene:5159', 'NCBIGene:6037', 'NCBIGene:339983', 'NCBIGene:146802', 'NCBIGene:673', 'NCBIGene:3791', 'NCBIGene:947', 'NCBIGene:760', 'NCBIGene:27', 'NCBIGene:9429', 'NCBIGene:1558', 'NCBIGene:81608', 'NCBIGene:765', 'NCBIGene:768', 'NCBIGene:1551', 'NCBIGene:6582', 'NCBIGene:1559', 'NCBIGene:5005', 'NCBIGene:1436', 'NCBIGene:25', 'NCBIGene:613', 'NCBIGene:6580', 'NCBIGene:761', 'NCBIGene:8647', 'NCBIGene:89845', 'NCBIGene:5004', 'NCBIGene:55244', 'NCBIGene:780', 'NCBIGene:21', 'NCBIGene:1326', 'NCBIGene:1577', 'NCBIGene:4921', 'NCBIGene:4914', 'NCBIGene:374569', 'NCBIGene:2322', 'NCBIGene:1565', 'NCBIGene:2064', 'NCBIGene:5742', 'NCBIGene:759', 'NCBIGene:771', 'NCBIGene:213', 'NCBIGene:6714', 'NCBIGene:766', 'NCBIGene:6916', 'NCBIGene:3815', 'NCBIGene:5156', 'NCBIGene:1576', 'NCBIGene:340527', 'NCBIGene:4835', 'NCBIGene:6532', 'NCBIGene:60482', 'NCBIGene:5243', 'NCBIGene:23632']
sql_in_select = "select node_code, ontology_id from comb_node_ontology where ontology_id in ({})"
sql_select_genes = "select ontology_id from comb_node_ontology where node_type_id = 2 limit {}"
DB_PASSWD = os.environ.get('DB_PASSWD')
number_genes = 20000


# methods
def build_in_statement(select_sql, list_input, debug=False):
    ''' builds the sql in statement '''
    in_sql = None

    # build the sql
    in_sql = ", ".join(list(map(lambda item: '%s', list_input)))
    if debug:
        logger.info("got sql '{}'".format(sql_in_select))

    # add in the in statement to the select sql
    new_sql = select_sql.format(in_sql)

    # return
    return new_sql

# main
if __name__ == "__main__":
    # get the connection
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db='tran_test_202108')
    cursor = conn.cursor()

    # select the gene list
    cursor.execute(sql_select_genes.format(number_genes))
    list_genes = cursor.fetchall()

    # build the sql
    sql = build_in_statement(sql_in_select ,list_genes)
    logger.info("got sql '{}'".format(sql))

    # search fr genes
    cursor.execute(sql, list_genes)
    results = cursor.fetchall()

    # print
    count = 0
    for row in results:
        count += 1
        if count % 100 == 0:
            logger.info("got name: {} and curie: {}".format(row[0], row[1]))
    logger.info("result size: {}".format(len(results)))


