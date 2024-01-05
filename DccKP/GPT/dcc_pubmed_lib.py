
# imports
import os
import pymysql as mdb
import xmltodict
import json
from time import gmtime, strftime
from random import shuffle

# constants
DB_PASSWD = os.environ.get('DB_PASSWD')
SCHEMA_GPT = "pubmed_gen"


# constants
SQL_SELECT_FILES_PROCESSED_BY_PROCESS_NAME = "select file_name from pubm_file_processed where process_name = %s"
SQL_INSERT_FILE_PROCESSED = "insert into pubm_file_processed (file_name, process_name) values(%s, %s)"
SQL_SELECT_ABSTRACT_BY_PUBMED_ID = "select pubmed_id from pubm_paper_abstract where pubmed_id = %s"
SQL_INSERT_ABSTRACT = "insert into pubm_paper_abstract (pubmed_id, abstract, title, journal_name, paper_year, in_pubmed_file) values(%s, %s, %s, %s, %s, %s)"

# methods
def get_connection(schema=SCHEMA_GPT):
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=schema)

    # return
    return conn

def get_db_if_pubmed_downloaded_general(conn, pubmed_id, log=False):
    '''
    will return value if abstract already downloaded, None otherwise
    '''
    # initialize
    result_id = None
    cursor = conn.cursor()

    # pick query 
    sql_select = SQL_SELECT_ABSTRACT_BY_PUBMED_ID

    # log
    if log:
        print("looking for abstract pubmed_id: {}".format(pubmed_id))

    # find
    cursor.execute(sql_select, (pubmed_id))
    db_result = cursor.fetchall()
    if db_result:
        result_id = db_result[0][0]
    else:
        if log:
            print("found abstract not downloaded for pubmed id: {}".format(pubmed_id))

    # return 
    return result_id

def insert_db_paper_abstract_general(conn, pubmed_id, abstract, title, journal, year, file_name=None, log=False):
    '''
    will insert a row int he paper abstract table
    '''
    # initialize
    cursor = conn.cursor()

    # log
    if log:
        print("inserting abstract for pubmed id: {} - for file: {}".format(pubmed_id, file_name))

    # shorten abstract if need be
    if abstract and len(abstract) > 3950:
        abstract = abstract[:3950]
    elif abstract:
        abstract = abstract.strip()

    # insert if data
    try:
        cursor.execute(SQL_INSERT_ABSTRACT, (pubmed_id, abstract, title, journal, year, file_name))
        conn.commit()
    except mdb.err.DataError:
        print("GOT DATABASE ERROR: skipping empty abstract for pubmed id: {}".format(pubmed_id))


def get_files_processed_list(conn, process_name, log=False):
    '''
    will return list of files processed for the process name
    '''
    # initialize
    list_files = []
    cursor = conn.cursor()

    # pick query 
    sql_select = SQL_SELECT_FILES_PROCESSED_BY_PROCESS_NAME

    # log
    if log:
        print("looking for process name: {}".format(process_name))

    # find
    cursor.execute(sql_select, (process_name))
    db_result = cursor.fetchall()
    for row in db_result:
        list_files.append(row[0])

    # return 
    return list_files

def insert_db_file_processed_general(conn, file_name, process_name, log=False):
    '''
    will insert a row in the file processed table
    '''
    # initialize
    cursor = conn.cursor()

    # log
    if log:
        print("inserting file processed: {} for process: {}".format(file_name, process_name))

    # insert if data
    try:
        cursor.execute(SQL_INSERT_FILE_PROCESSED, (file_name, process_name))
        conn.commit()
    except mdb.err.DataError:
        print("GOT DATABASE ERROR: skipping file name: {}".format(file_name))
