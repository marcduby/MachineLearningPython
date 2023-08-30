
# imports
import os 
import xml.etree.ElementTree as ET
import xmltodict
import re
import glob 
import requests
import io
import json
import xml
import time
import pymysql as mdb

# for AWS
ENV_DIR_CODE = os.environ.get('DIR_CODE')
ENV_DIR_PUBMED = os.environ.get('DIR_PUBMED')

# import relative libraries
dir_code = "/home/javaprog/Code/PythonWorkspace/"
if ENV_DIR_CODE:
    dir_code = ENV_DIR_CODE
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/GPT/')
import dcc_gpt_lib

# constants
URL_SEARCH_BY_KEYWORD = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={}"
URL_SEARCH_BY_WEB_ENV = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?query_key={}&WebEnv={}&cmd=neighbor_history"

# sql constants
DB_PASSWD = os.environ.get('DB_PASSWD')
SCHEMA_GPT = "gene_gpt"
DB_PAPER_TABLE = "pgpt_paper"
DB_PAPER_ABSTRACT = "pgpt_paper_abtract"

SQL_INSERT_PUBMED = "insert into {}.pgpt_paper (pubmed_id) values(%s)".format(SCHEMA_GPT)
SQL_SELECT_PUBMED = "select pubmed_id from {}.pgpt_paper where pubmed_id = %s".format(SCHEMA_GPT)

SQL_SELECT_SEARCH_PUBMED = "select pubmed_id from {}.pgpt_search_paper where search_id = %s and pubmed_id = %s".format(SCHEMA_GPT)
SQL_INSERT_SEARCH_PUBMED = "insert into {}.pgpt_search_paper (search_id, pubmed_id) values(%s, %s)".format(SCHEMA_GPT)
SQL_UPDATE_SEARCH_COUNT = "update {}.pgpt_search set pubmed_count = %s where id = %s ".format(SCHEMA_GPT)
SQL_SELECT_SEARCH_LIST_TO_DOWNLOAD = "select id, terms from {}.pgpt_search where to_download_ids = 'Y' order by id desc ".format(SCHEMA_GPT)
SQL_UPDATE_SEARCH_AFTER_DOWNLOAD = "update {}.pgpt_search set to_download_ids = 'N' where id = %s ".format(SCHEMA_GPT)


# methods
def get_pubmed_ids_query_key(list_keywords, use_history=False, log=False):
    '''
    will query the pubmed rest service and will retrieve the pubmed id list related to the provided keywords
    '''
    # initialize
    list_pubmed_ids = []
    web_env = None
    query_key = None
    count_pubmed = -1

    # build keywords
    input_keyword = ",".join(list_keywords)

    # query
    url_string = URL_SEARCH_BY_KEYWORD.format(input_keyword)
    if use_history:
        url_string = url_string + '&usehistory=y'
    if log:
        print("got request: {}".format(url_string))

    # get the pubmed ids
    response = requests.get(url_string)
    map_response = xmltodict.parse(response.content)
    if log:
        print("got rest response: {}".format(map_response))

    list_response_id = map_response.get('eSearchResult').get('IdList')
    if list_response_id:
        if list_response_id.get('Id'):
            for item in list_response_id.get('Id'):
                list_pubmed_ids.append(item)

    # get history 
    if map_response.get('eSearchResult'):
        web_env = map_response.get('eSearchResult').get('WebEnv')
        query_key = map_response.get('eSearchResult').get('QueryKey')
        count_pubmed = map_response.get('eSearchResult').get('Count')
        if count_pubmed:
            count_pubmed = int(count_pubmed)

    # log
    if log:
        print("got pubmed id list: {}".format(list_pubmed_ids))

    # return
    return list_pubmed_ids, web_env, query_key, count_pubmed

def get_list_pubmed_ids_from_webenv(web_env, query_key, log=False):
    '''
    retrieves the abstracts from all pubmed articles linked to the provided web env and query key
    '''
    # initialize
    list_response_id = []
    
    # query for the web env and query key
    if web_env and query_key:
        url_string = URL_SEARCH_BY_WEB_ENV.format(query_key, web_env)
        if log:
            print("\nquery with url: {}".format(url_string))

        # query
        response = requests.get(url_string)
        map_response = xmltodict.parse(response.content)
        if log:
            print("got rest response: {}".format(map_response))

        list_response_id = map_response.get('eLinkResult').get('LinkSet').get('IdList').get('Id')

        # if only one element, create list
        if isinstance(list_response_id, str) or isinstance(list_response_id, int):
            list_response_id = [list_response_id]


    # return
    return list_response_id

def get_db_pubmed_id(conn, pubmed_id, log=False):
    '''
    find keyword PK or return None
    '''
    paper_id = None
    cursor = conn.cursor()

    # pick query 
    sql_select = SQL_SELECT_PUBMED

    # find
    cursor.execute(sql_select, (pubmed_id))
    db_result = cursor.fetchall()
    if db_result:
        paper_id = db_result[0][0]

    # return 
    return paper_id

def insert_db_pubmed_id(conn, pubmed_id, log=False):
    '''
    will insert a pubmed id
    '''
    # initialize
    result_id = None
    cursor = conn.cursor()
    int_pubmed = int(pubmed_id)

    # see if already in db
    result_id = get_db_pubmed_id(conn, pubmed_id)

    # if not, insert
    if not result_id:
        if log:
            print("inserting into table pubmed_id: {}".format(int_pubmed))
        cursor.execute(SQL_INSERT_PUBMED, (int_pubmed))
        conn.commit()

def get_db_pubmed_id_for_search(conn, search_id, pubmed_id, log=False):
    '''
    find keyword PK or return None
    '''
    paper_id = None
    cursor = conn.cursor()

    # pick query 
    sql_select = SQL_SELECT_SEARCH_PUBMED

    # find
    cursor.execute(sql_select, (search_id, pubmed_id))
    db_result = cursor.fetchall()
    if db_result:
        paper_id = db_result[0][0]

    # return 
    return paper_id

def insert_db_pubmed_id_for_search(conn, search_id, pubmed_id, log=False):
    '''
    will insert a pubmed id search/pubmed link table
    '''
    # initialize
    result_id = None
    cursor = conn.cursor()
    int_pubmed = int(pubmed_id)

    # see if already in db
    result_id = get_db_pubmed_id_for_search(conn, search_id, pubmed_id)

    # if not, insert
    if not result_id:
        if log:
            print("inserting search: {}, pubmed_id: {}".format(search_id, int_pubmed))
        cursor.execute(SQL_INSERT_SEARCH_PUBMED, (search_id, int_pubmed))
        conn.commit()

def update_db_search_pubmed_count(conn, search_id, count_pubmed, log=False):
    '''
    will update the search to done
    '''
    # initialize
    cursor = conn.cursor()

    # log
    if log:
        print("setting pubmed count to: {} for search: {}".format(count_pubmed, search_id))

    # see if already in db
    if count_pubmed is not None:
        cursor.execute(SQL_UPDATE_SEARCH_COUNT, (count_pubmed, search_id))
        conn.commit()
    else:
        print("ERROR: got none count for pubmed for search: {}".format(count_pubmed))

def get_db_pubmed_searches_list(conn, log=False):
    '''
    find all the pubmed searches to do
    '''
    list_search = []
    cursor = conn.cursor()

    # pick query 
    # find
    cursor.execute(SQL_SELECT_SEARCH_LIST_TO_DOWNLOAD)
    db_result = cursor.fetchall()
    for row in db_result:
        search_id = row[0]
        search_terms = row[1]
        list_search.append({'id': search_id, 'terms': search_terms})

    # return 
    return list_search

def update_db_search_to_done(conn, search_id, log=False):
    '''
    will update the search to done
    '''
    # initialize
    cursor = conn.cursor()

    # see if already in db
    cursor.execute(SQL_UPDATE_SEARCH_AFTER_DOWNLOAD, (search_id))
    conn.commit()

def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=SCHEMA_GPT)

    # return
    return conn


# main
if __name__ == "__main__":
    # initialize
    conn = get_connection()

    # DEBUG
    # id_pubmed_test = 34705354
    # text_test = get_pubmed_abtract(id_pubmed=id_pubmed_test, log=True)

    # get list of searches
    list_searches = get_db_pubmed_searches_list(conn)
    # list_searches=[{'id': 9, 'terms': 'GCK,human'}]
    # list_searches=[{'id': 79, 'terms': 'CNTD1,human'}] - test for single result which was splitting strng into characters in for loop

    # loop
    for item in list_searches:
        # get search terms
        print("\n=======search pubmed for: {}".format(item.get('terms')))
        list_keywords = item.get('terms').split(",")
        search_id = item.get('id')

        try :
            # query for pubmed ids
            list_pubmed_ids, web_env, query_key, count_pubmed = get_pubmed_ids_query_key(list_keywords=list_keywords, use_history=True, log=True)
            print("for {}, got original list of papers of size: {}\n\n".format(list_keywords, count_pubmed))

            # update the search count
            update_db_search_pubmed_count(conn=conn, search_id=search_id, count_pubmed=count_pubmed, log=True)

            # get all the pubmed ids
            # web_env = "MCID_64b697f1a865fa461a518090"
            # query_key = 1
            # time.sleep(90)
            if count_pubmed  and count_pubmed > 0:
                # time.sleep(30)
                time.sleep(10)
                # time.sleep(5)
                print("query pubmed for queryKey: {} and webEnv: {}".format(query_key, web_env))
                list_pubmed_ids = get_list_pubmed_ids_from_webenv(web_env, query_key, log=False)
                print("\ngot list of pubmed ids of size: {}".format(len(list_pubmed_ids)))

                # insert pubmed ids
                for pubmed_id in list_pubmed_ids:
                    insert_db_pubmed_id_for_search(conn, search_id=search_id, pubmed_id=pubmed_id, log=True)
                    insert_db_pubmed_id(conn, pubmed_id=pubmed_id, log=True)

            # update search to done
            update_db_search_to_done(conn=conn, search_id=search_id)
            print("setting search: {} to done with pubmed count: {}".format(search_id, count_pubmed))

            # # pause for next query
            # time.sleep(10)

        except xml.parsers.expat.ExpatError:
            print("got xml parsing error; continue\n\n")
            # pause for next query
            time.sleep(20)
            continue

        except requests.exceptions.ChunkedEncodingError:
            print("got requests error; continue\n\n")
            # pause for next query
            time.sleep(20)
            continue

        # # pause for next query
        # time.sleep(90)
