
# imports
import os 
import pymysql as mdb
import glob 
import io
import json

# constants
DB_PASSWD = os.environ.get('DB_PASSWD')
SCHEMA_GPT = "pubmed_gpt"
DIR_DATA = "/home/javaprog/Data/Broad/GPT/Data/ConvoPubmedV1"
# FILE_DATA = "{}/text_generation_keywords_train_60k.json".format(DIR_DATA)
FILE_DATA = "{}/text_generation_keywords_train_chem_100k.json".format(DIR_DATA)
SQL_SELECT = "select keyword from {}.pmd_keyword".format(SCHEMA_GPT)
SQL_WHERE = " where abstract_text like %s"
SQL_WHERE = " where pubmed_id in (36061186,35928446,36072671,36171883,36173399,35910211,36105085,35754818,35480303)"
LIST_REPLACE = [["\n", ""], ["CI.", "CI"], []]

# methods
def create_json_keyword_file(list_input, file_name, log=False):
    # save to json
    with open(file_name, "w+") as f:
        json.dump(list_input, f)

    print("wrote out: {} size list to: {}".format(len(list_input), file_name))

def get_list_of_keywords(conn, log=False):
    '''
    retrieves the list of abtsracts from the database
    '''
    cursor = conn.cursor()
    list_keywords = []

    cursor.execute(SQL_SELECT, ())

    # get the results
    db_results = cursor.fetchall()
    for row in db_results:
        # list_keywords.append(row[0])
        list_keywords = list_keywords + row[0].split(" ")

    # return
    return list_keywords

def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=SCHEMA_GPT)

    # return
    return conn


# main
if __name__ == "__main__":
    # initalize
    list_keywords = []

    # get the list of abstracts
    conn = get_connection()
    # list_abstracts = get_list_of_abstracts(conn, 'PCSK9')
    list_keywords = get_list_of_keywords(conn)
    print("to process, got list of keywords of size: {}".format(len(list_keywords)))

    # write out the conversations
    create_json_keyword_file(list_keywords, FILE_DATA)