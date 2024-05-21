
# impports
import os 
import pymysql as mdb
import glob 
import io
import json

# constants
DB_PASSWD = os.environ.get('DB_PASSWD')
SCHEMA_GPT = "pubmed_gpt"
DIR_DATA = "/home/javaprog/Data/Broad/GPT/Data/ConvoPubmedV1"
FILE_DATA = "{}/text_generation_data_train_chem_100k.json".format(DIR_DATA)
SQL_SELECT = "select pubmed_id, abstract_text from {}.pmd_abstract".format(SCHEMA_GPT)
SQL_WHERE = " where abstract_text like %s"
SQL_WHERE = " where pubmed_id in (36061186,35928446,36072671,36171883,36173399,35910211,36105085,35754818,35480303)"
LIST_REPLACE = [["\n", ""], ["CI.", "CI"], []]
STR_PROMPT = "for the following, summarize the biology of gene {}:\n{}"


def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=SCHEMA_GPT)

    # return
    return conn

def create_concat_abstracts(list_abstracts, gene, size_token=4000, log=False):
    '''
    creates a sized concatenated abstract from the given list
    ''' 
    result_abstract = ""

    # loop
    for abstract in list_abstracts:
        temp_abstract = result_abstract + abstract
        if len(temp_abstract.split(" ")) < size_token:
            result_abstract = temp_abstract
        else:
            break

    # return
    return result_abstract

def get_list_abstracts(conn, keyword, log=False):
    '''
    get a list of abstracts with the keyword in them 
    '''
    sql_select = "select abstract_text from pmd_abstract where lower(abstract_text) like lower(%s)"
    list_abstracts = []

    # get the cursor
    cursor = conn.cursor()

    # query 
    cursor.execute(sql_select, ('%{}%'.format(keyword)))
    list_results = cursor.fetchall()
    print("got sql list of size: {}".format(len(list_results)))

    # loop
    for row in list_results:
        list_abstracts.append(row[0])

    # log
    # print(list_results)

    # return
    return list_abstracts


# main
if __name__ == "__main__":
    # initialize
    gene = "UBE2NL"
    gene = "SLC30A8"
    gene = "MAP3K15"
    gene = "GIGYF1"
    gene = "GPR75"
    gene = "INHBE"

    # get the connection
    conn = get_connection()

    # get the abstracts
    list_abstracts = get_list_abstracts(conn, gene)
    print("got abstract list of size: {}".format(len(list_abstracts)))

    # get the concatenated abstract
    abstract_concat = create_concat_abstracts(list_abstracts, gene)

    # print gpt prompt
    str_final = STR_PROMPT.format(gene, abstract_concat)
    print(str_final)
