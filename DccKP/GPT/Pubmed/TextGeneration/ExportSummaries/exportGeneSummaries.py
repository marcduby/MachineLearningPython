

# imports
import os
import json
import pymysql as mdb

# constants
ENV_DB_USER = os.environ.get('DB_USER')
ENV_DB_PASSWD = os.environ.get('DB_PASSWD')
SCHEMA_GPT = "web_gpt"
SQL_SELECT_GENE_ABSTRACT = "select gene_code, abstract from gene_abstract order by gene_code"
FILE_RESULTS = "/home/javaprog/Data/Broad/dig-analysis-data/gene_gpt/gene_abstracts_gpt.json"
DB_PASSWD = os.environ.get('DB_PASSWD')

# methods
def get_list_gene_summaries(conn, log=False):
    '''
    returns a list of gene summaries
    '''
    list_genes = []
    cursor = conn.cursor()

    # get the data
    cursor.execute(SQL_SELECT_GENE_ABSTRACT)

    # query 
    db_result = cursor.fetchall()
    for row in db_result:
        gene = row[0]
        abstract = row[1]
        list_genes.append({"gene": gene, 'abstract': abstract})

    # return
    return list_genes

def get_connection(schema=SCHEMA_GPT):
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=schema)

    # return
    return conn


# main
if __name__ == "__main__":
    # get the connection
    conn = get_connection()

    # get the list of gene abstracts
    list_abstracts = get_list_gene_summaries(conn=conn)
    print("got number of gene abstracts: {}".format(len(list_abstracts)))

    # write out the file
    with open(FILE_RESULTS, 'w') as file:
        json.dump(list_abstracts, file, indent=2)

          

