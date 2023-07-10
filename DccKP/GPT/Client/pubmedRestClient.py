
# imports
import os 
import xml.etree.ElementTree as ET
import xmltodict
import re
import glob 
import requests
import io


# constants
URL_PAPER = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={}"
URL_SEARCH_BY_KEYWORD = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={}"

# methods
def get_pubmed_ids(list_keywords, log=False):
    '''
    will query the pubmed rest service and will retrieve the pubmed id list related to the provided keywords
    '''
    # initialize
    list_pubmed_ids = []

    # build keywords
    input_keyword = "&".join(list_keywords)

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

    # return
    return list_pubmed_ids    



# main
if __name__ == "__main__":
    # set keywords
    list_keywords = ['MAP3K15', 'diabetes']

    # query for pubmed ids
    list_pubmed_ids = get_pubmed_ids(list_keywords, log=True)
    print("for {}, got list of papers of size: {}".format(list_keywords, len(list_pubmed_ids)))



