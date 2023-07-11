
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

# constants
URL_PAPER = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={}"
URL_SEARCH_BY_KEYWORD = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={}"

ID_TEST_ARTICLE = 32163230
ID_TEST_ARTICLE = 36383675
ID_TEST_ARTICLE = 37303064

# methods
def get_pubmed_ids(list_keywords, log=False):
    '''
    will query the pubmed rest service and will retrieve the pubmed id list related to the provided keywords
    '''
    # initialize
    list_pubmed_ids = []

    # build keywords
    input_keyword = ",".join(list_keywords)

    # query
    url_string = URL_SEARCH_BY_KEYWORD.format(input_keyword)
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
    return list_pubmed_ids    

def get_pubmed_abtract(id_pubmed, log=False):
    '''
    get the pubmed abstrcat for the given id
    '''
    # initialize
    list_temp = []
    text_abstract = ""

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
    
    # return
    return text_abstract

def get_list_abstracts(list_keywords, log=False):
    '''
    retrieves the abstrcats from all pubmed articles linked to the provided keywords
    '''
    # initialize
    list_abstracts = []

    # get the ids
    list_pubmed_ids = get_pubmed_ids(list_keywords, log=log)

    # get the abstracts
    for item in list_pubmed_ids:
        time.sleep(3)
        print("getting abstract for: {}".format(item))
        text_abstract = get_pubmed_abtract(item, log=log)
        list_abstracts.append(text_abstract)

    # return
    return list_abstracts

# main
if __name__ == "__main__":
    # set keywords
    list_keywords = ['MAP3K15', 'diabetes']

    # query for pubmed ids
    list_pubmed_ids = get_pubmed_ids(list_keywords, log=True)
    print("for {}, got list of papers of size: {}".format(list_keywords, len(list_pubmed_ids)))

    # query for the abstract text
    text_abstract = get_pubmed_abtract(ID_TEST_ARTICLE, log=False)
    print("got abstract: \n{}".format(text_abstract))

    # get the abstract list
    list_abstracts = get_list_abstracts(list_keywords, log=True)
    for item in list_abstracts:
        print("abstract: \n{}\n".format(item))



