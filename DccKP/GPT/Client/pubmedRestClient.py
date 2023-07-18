
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
DB_PASSWD = os.environ.get('DB_PASSWD')
SCHEMA_GPT = "gene_gpt"
DB_PAPER_TABLE = "pgpt_paper"
DB_PAPER_ABSTRACT = "pgpt_paper_abtract"
URL_PAPER = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={}"
URL_SEARCH_BY_KEYWORD = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={}"
URL_SEARCH_BY_WEB_ENV = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?query_key={}&WebEnv={}&cmd=neighbor_history"

ID_TEST_ARTICLE = 32163230
ID_TEST_ARTICLE = 36383675
ID_TEST_ARTICLE = 37303064

LIST_KEYWORDS = ['UBE2NL', 'human']
LIST_KEYWORDS = ['UBE2NL']
LIST_KEYWORDS = ['PPARG', 'human']

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

    # build keywords
    input_keyword = ",".join(list_keywords)

    # query
    url_string = URL_SEARCH_BY_KEYWORD.format(input_keyword)
    if use_history:
        url_string = url_string + 'usehistory=y'
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
    # log
    if log:
        print("got pubmed id list: {}".format(list_pubmed_ids))

    # return
    return list_pubmed_ids, web_env, query_key

def get_pubmed_abtract(id_pubmed, log=False):
    '''
    get the pubmed abstrcat for the given id
    '''
    # initialize
    list_temp = []
    text_abstract = ""
    title = ""
    journal = ""
    year = ""

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

    list_abstract = map_response.get('PubmedArticleSet').get('PubmedArticle').get('MedlineCitation').get('Article').get('Abstract').get('AbstractText')
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
    year = map_response.get('PubmedArticleSet').get('PubmedArticle').get('MedlineCitation').get('DateCompleted').get('Year')
    title = map_response.get('PubmedArticleSet').get('PubmedArticle').get('MedlineCitation').get('Article').get('ArticleTitle')
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

def insert_pubmed_id(conn, pubmed_id, log=False):
    '''
    will insert a pubmed id in the database if necessary
    '''
    # initialize

    # see if already in db

    # if not, insert



# main
if __name__ == "__main__":
    # set keywords
    list_keywords = ['MAP3K15', 'human']
    list_keywords = ['PPARG', 'human']
    list_keywords = LIST_KEYWORDS

    # query for pubmed ids
    list_pubmed_ids, web_env, query_key = get_pubmed_ids_query_key(list_keywords, log=True)
    print("for {}, got original list of papers of size: {}\n\n".format(list_keywords, len(list_pubmed_ids)))

    # query for the abstract text
    text_abstract, title, journal, year = get_pubmed_abtract(ID_TEST_ARTICLE, log=True)
    print("got abstract: \n{}".format(text_abstract))
    print("title: {} for journal: {} and year: {}".format(title, journal, year))

    # # get the abstract list
    # list_abstracts = get_list_abstracts(list_keywords, log=True)
    # for item in list_abstracts:
    #     print("abstract: \n{}\n".format(item))

    # # get the list of chat inputs
    # print("\ngettign chat inputs")
    # list_chat_inputs = get_list_gpt_inputs(list_abstracts)
    # for item in list_chat_inputs:
    #     print("\ngpt input: \n{}\n".format(item))


    # get the abstract list
    web_env = "MCID_64b697f1a865fa461a518090"
    query_key = 1
    list_pubmed_ids = get_list_pubmed_ids_from_webenv(web_env, query_key, log=False)
    print("\ngot list of pubmed ids of size: {}".format(len(list_pubmed_ids)))

