
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

# constants

# constants
URL_PAPER = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={}"
URL_SEARCH_BY_KEYWORD = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={}"
URL_SEARCH_BY_WEB_ENV = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?query_key={}&WebEnv={}&cmd=neighbor_history"

ID_TEST_ARTICLE = 32163230
ID_TEST_ARTICLE = 36383675
ID_TEST_ARTICLE = 37303064

LIST_KEYWORDS = ['UBE2NL', 'human']
LIST_KEYWORDS = ['UBE2NL']
LIST_KEYWORDS = ['PPARG', 'human']

# sql constants
DB_PASSWD = os.environ.get('DB_PASSWD')
SCHEMA_GPT = "gene_gpt"
DB_PAPER_TABLE = "pgpt_paper"
DB_PAPER_ABSTRACT = "pgpt_paper_abtract"

SQL_SELECT_PUBMED = "select pubmed_id from {}.pgpt_paper where pubmed_id = %s".format(SCHEMA_GPT)
SQL_SELECT_ABSTRACT = "select pubmed_id from {}.pgpt_paper_abstract where pubmed_id = %s".format(SCHEMA_GPT)

SQL_INSERT_PAPER = "insert into {}.pgpt_paper (pubmed_id) values(%s)".format(SCHEMA_GPT)
SQL_INSERT_ABSTRACT = "insert into {}.pgpt_paper_abstract (pubmed_id, abstract, title, journal_name, paper_year, document_level) values(%s, %s, %s, %s, %s, %s)".format(SCHEMA_GPT)

SQL_SELECT_SEARCH_PAPER = "select id from {}.pgpt_search_paper where search_id = %s and paper_id = %s".format(SCHEMA_GPT)
SQL_INSERT_SEARCH_PAPER = "insert into {}.pgpt_search_paper (search_id, paper_id) values(%s, %s)".format(SCHEMA_GPT)

# SQL_SELECT_SEARCH_LIST = "select id, terms from {}.pgpt_search order by id desc".format(SCHEMA_GPT)
SQL_SELECT_SEARCH_LIST_TO_DOWNLOAD = "select id, terms from {}.pgpt_search where to_download = 'Y' order by id ".format(SCHEMA_GPT)
SQL_UPDATE_SEARCH_AFTER_DOWNLOAD = "update {}.pgpt_search set to_download = 'N', date_last_download = sysdate() where id = %s ".format(SCHEMA_GPT)
SQL_UPDATE_SEARCH_COUNT = "update {}.pgpt_search set pubmed_count = %s where id = %s ".format(SCHEMA_GPT)


# methods
def get_list_gpt_inputs(list_abstracts, num_limit=4096, log=True):
    '''
    returns list of combined inputs less than size given
    '''
    list_chat_inputs = []
    temp_sentence = ""

    # loop
    for item in list_abstracts:
        # make sure new sentences doesn't exceed the limit
        if len(temp_sentence.split()) + len(item.split()):
            temp_sentence = temp_sentence + " " + item
        else:
            list_chat_inputs.append(temp_sentence)
            temp_sentence = item

    # append the last one    
    list_chat_inputs.append(temp_sentence)

    # return
    return list_chat_inputs


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

    list_response_id = map_response.get('eSearchResult').get('IdList').get('Id')
    if list_response_id:
        for item in list_response_id:
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

def get_pubmed_abtract(id_pubmed, log=False):
    '''
    get the pubmed abstrcat for the given id
    '''
    # initialize
    list_temp = []
    text_abstract = ""
    title = ""
    journal = ""
    year = 0

    # sleep
    time.sleep(5)

    # get the url
    url_string = URL_PAPER.format(id_pubmed)

    if log:
        print("got url: {}".format(url_string))

    # get the abstract
    response = requests.get(url_string)
    try:
        map_response = xmltodict.parse(response.content)
    except xml.parsers.expat.ExpatError:
        print("GOT ERROR: {}".format(response.content))

    if log:
        print("got rest response: {}".format(json.dumps(map_response, indent=1)))

    if map_response.get('PubmedArticleSet') and map_response.get('PubmedArticleSet').get('PubmedArticle') and map_response.get('PubmedArticleSet').get('PubmedArticle').get('MedlineCitation'):
        resp_article = map_response.get('PubmedArticleSet').get('PubmedArticle').get('MedlineCitation').get('Article')
        if resp_article and resp_article.get('Abstract'):
            list_abstract = resp_article.get('Abstract').get('AbstractText')
            if log:
                print(list_abstract)

            if list_abstract:
                if isinstance(list_abstract, list):
                    for item in list_abstract:
                        list_temp.append(item.get('#text'))
                    text_abstract = " ".join(list_temp)
                elif isinstance(list_abstract, dict):
                    text_abstract = list_abstract.get('#text')
                elif isinstance(list_abstract, str):
                    text_abstract = list_abstract

            # get year
            if map_response.get('PubmedArticleSet').get('PubmedArticle').get('MedlineCitation').get('DateCompleted'):
                year = map_response.get('PubmedArticleSet').get('PubmedArticle').get('MedlineCitation').get('DateCompleted').get('Year')
            title = map_response.get('PubmedArticleSet').get('PubmedArticle').get('MedlineCitation').get('Article').get('ArticleTitle')
            if isinstance(title, dict):
                title = title.get('#text')
            journal = map_response.get('PubmedArticleSet').get('PubmedArticle').get('MedlineCitation').get('Article').get('Journal').get('Title')

    # return
    return text_abstract, title, journal, year

def get_list_abstracts(list_keywords, log=False):
    '''
    retrieves the abstrcats from all pubmed articles linked to the provided keywords
    '''
    # initialize
    list_abstracts = []

    # get the ids
    list_pubmed_ids, web_env, query_key = get_pubmed_ids_query_key(list_keywords, log=log)

    # get the abstracts
    for item in list_pubmed_ids:
        time.sleep(2)
        print("getting abstract for: {}".format(item))
        text_abstract = get_pubmed_abtract(item, log=log)
        list_abstracts.append(text_abstract)

    # return
    return list_abstracts

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

    # return
    return list_response_id

def get_db_pubmed_id(conn, pubmed_id, look_for_abstract=False, log=False):
    '''
    find keyword PK or return None
    '''
    paper_id = None
    cursor = conn.cursor()

    # pick query 
    if look_for_abstract:
        sql_select = SQL_SELECT_ABSTRACT
    else:
        sql_select = SQL_SELECT_PUBMED

    # find
    cursor.execute(sql_select, (pubmed_id))
    db_result = cursor.fetchall()
    if db_result:
        paper_id = db_result[0][0]

    # return 
    return paper_id

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

def insert_pubmed_id(conn, pubmed_id, abstract=None, title=None, journal=None, year=None, in_abstract=False, log=False):
    '''
    will insert a pubmed id in the database if necessary
    '''
    # initialize
    result_id = None
    cursor = conn.cursor()
    int_pubmed = int(pubmed_id)

    # see if already in db
    result_id = get_db_pubmed_id(conn, pubmed_id, look_for_abstract=in_abstract)

    # shorten abstract if need be
    if abstract and len(abstract) > 3950:
        abstract = abstract[:3950]

    # if not, insert
    if not result_id:
        if in_abstract:
            if log:
                print("inserting: {} - {} - {}".format(year, journal, title))
            int_year = int(year)
            cursor.execute(SQL_INSERT_ABSTRACT, (int_pubmed, abstract, title, journal, int_year, 0))
        else:
            cursor.execute(SQL_INSERT_PAPER, (int_pubmed))
        conn.commit()

def insert_db_search_paper(conn, pubmed_id, search_id, log=False):
    '''
    will insert a paper/search link in the db
    '''
    # initialize
    result_id = None
    cursor = conn.cursor()
    int_pubmed = int(pubmed_id)

    # see if already in db
    cursor.execute(SQL_SELECT_SEARCH_PAPER, (search_id, int_pubmed))
    db_results = cursor.fetchall()

    # if not, insert
    if not db_results:
        cursor.execute(SQL_INSERT_SEARCH_PAPER, (search_id, int_pubmed))
        conn.commit()

def update_db_search_to_done(conn, search_id, log=False):
    '''
    will update the search to done
    '''
    # initialize
    cursor = conn.cursor()

    # see if already in db
    cursor.execute(SQL_UPDATE_SEARCH_AFTER_DOWNLOAD, (search_id))
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

    # get list of searches
    list_searches = get_db_pubmed_searches_list(conn)
    # list_searches=[{'id': 9, 'terms': 'GCK,human'}]

    # loop
    for item in list_searches:
        # get search terms
        print("searh pubmed for: {}".format(item.get('terms')))
        list_keywords = item.get('terms').split(",")
        search_id = item.get('id')

        # query for pubmed ids
        list_pubmed_ids, web_env, query_key, count_pubmed = get_pubmed_ids_query_key(list_keywords=list_keywords, use_history=True, log=True)
        print("for {}, got original list of papers of size: {}\n\n".format(list_keywords, len(list_pubmed_ids)))

        # update the search count
        update_db_search_pubmed_count(conn=conn, search_id=search_id, count_pubmed=count_pubmed, log=True)

        # get the abstract list
        # web_env = "MCID_64b697f1a865fa461a518090"
        # query_key = 1
        print("query pubmed for queryKey: {} and webEnv: {}".format(query_key, web_env))
        list_pubmed_ids = get_list_pubmed_ids_from_webenv(web_env, query_key, log=False)
        print("\ngot list of pubmed ids of size: {}".format(len(list_pubmed_ids)))

        # insert pubmed ids
        for pubmed_id in list_pubmed_ids:
        # for i in range(100000):
            # pubmed_id = list_pubmed_ids[i]

            # get the abstract
            if not get_db_pubmed_id(conn, pubmed_id, look_for_abstract=True):
                print("inserting journal abstract for pubmed id: {}".format(pubmed_id))
                text_abstract, title, journal, year = get_pubmed_abtract(pubmed_id, log=True)

                # insert
                if text_abstract and title and journal:
                    insert_pubmed_id(conn, pubmed_id)
                    insert_pubmed_id(conn, pubmed_id, text_abstract, title, journal, year, in_abstract=True, log=True)
                    insert_db_search_paper(conn=conn, pubmed_id=pubmed_id, search_id=search_id)
                else:
                    print("ERROR: skipping {} due to lack of data".format(pubmed_id))

        # update search when done
        update_db_search_to_done(conn=conn, search_id=search_id)

        # pause for next query
        time.sleep(120)







# for testing
    # # set keywords
    # list_keywords = ['MAP3K15', 'human']
    # list_keywords = ['PPARG', 'human']
    # list_keywords = LIST_KEYWORDS

    # # query for the abstract text
    # text_abstract, title, journal, year = get_pubmed_abtract(ID_TEST_ARTICLE, log=False)
    # print("got abstract: \n{}".format(text_abstract))
    # print("title: {} for journal: {} and year: {}".format(title, journal, year))

    # # get the abstract list
    # list_abstracts = get_list_abstracts(list_keywords, log=True)
    # for item in list_abstracts:
    #     print("abstract: \n{}\n".format(item))

    # # get the list of chat inputs
    # print("\ngettign chat inputs")
    # list_chat_inputs = get_list_gpt_inputs(list_abstracts)
    # for item in list_chat_inputs:
    #     print("\ngpt input: \n{}\n".format(item))

