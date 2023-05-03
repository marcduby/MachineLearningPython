

# imports
import os 
import pymysql as mdb
import xml.etree.ElementTree as ET
import xmltodict
import re
import glob 
import io

# constants 
COL_PUBMED = "pubmed_id"
COL_YEAR = "num_year"
COL_MONTH = "num_month"
COL_DAY = "num_day"
COL_TITLE = "paper_title"
COL_JOURNAL = "journal_title"
COL_ABSTRACT = "abstract_text"
COL_DATE = "paper_date"
COL_LINKS = "keyword_links"

FILE_TEST_XML = "/home/javaprog/Data/Broad/GPT/Pubmed/Load20230501/pubmed23n1007.xml"
TEXT_JOURNAL_SUBNAME = "genet"
DIR_XML = "/home/javaprog/Data/Broad/GPT/Pubmed/Load20230501/"

# methods
def parse_abstract_file(xmlfile, journal_substring=None, log=False):
    '''
    parses the abstract xml file, adds journal names to set
    '''
    set_results = set()
    xml_doc = None 

    # parse the file
    with open(xmlfile) as xml_input:
        xml_doc = xmltodict.parse(xml_input.read())

    # loop through the papers
    for paper in xml_doc.get('PubmedArticleSet').get('PubmedArticle'):
        # print(paper)
        # break
        # empty paper dictionary
        map_paper = {}

        # get the journal title
        journal_name = paper.get('MedlineCitation').get('Article').get('Journal').get('ISOAbbreviation')

        # add to list
        if journal_name:
            set_results.add(journal_name)

    # log
    print("returning set of size: {}".format(len(set_results)))

    # return 
    return set_results


# main
if __name__ == "__main__":
    # initialize
    set_journals = set()
    max_count = 5
    num_count = 0

    # get the list of input files
    files = [file for file in glob.glob(DIR_XML + "*.xml")]

    # for each file, get the set of journal names
    # files = [FILE_TEST_XML]
    for file_name in files:
        if num_count < max_count:
            print("reading file: {}".format(file_name))
            set_journals.update(parse_abstract_file(file_name))
            num_count = num_count + 1

    # sort the journal nales
    print("sorting set of size: {}".format(len(set_journals)))
    list_journals = list(set_journals)
    print("sorting list of size: {}".format(len(list_journals)))
    list_journals.sort()

    # print
    print("got list of journals of size: {}".format(len(list_journals)))
    for item in list_journals:
        print("name: {}".format(item))



