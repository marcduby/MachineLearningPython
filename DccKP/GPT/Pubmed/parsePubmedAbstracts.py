
import os 
import pymysql as mdb
import xml.etree.ElementTree as ET
import xmltodict
import re
import glob 
import io

# read all files, gram only the abstracts 
# for each abstract, loop through all translator db genes and tag articles that have gene in them 
# use text as input, gene list as labels to train GPT 

# tables 
# gene table 
# abstract table
# gene/abstract link table

# constants 
DB_PASSWD = os.environ.get('DB_PASSWD')
SCHEMA_GPT = "pubmed_gpt"
TABLE_ABSTRACT = "pmd_abstract"

COL_PUBMED = "pubmed_id"
COL_YEAR = "num_year"
COL_MONTH = "num_month"
COL_DAY = "num_day"
COL_TITLE = "paper_title"
COL_JOURNAL = "journal_title"
COL_ABSTRACT = "abstract_text"
COL_DATE = "paper_date"

SQL_INSERT = """
    insert into {}.{} (pubmed_id, paper_title, paper_date, journal_title, abstract_text)
    values(%s, %s, %s, %s, %s)""".format(SCHEMA_GPT, TABLE_ABSTRACT)
SQL_DELETE = "delete from {}.{}".format(SCHEMA_GPT, TABLE_ABSTRACT)

FILE_TEST_XML = "/home/javaprog/Code/PythonWorkspace/MachineLearningPython/DccKP/GPT/Pubmed/test.xml"
DIR_XML = "/home/javaprog/Data/Broad/GPT/Pubmed/"

def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=SCHEMA_GPT)

    # return
    return conn


def parse_abstract_file(xmlfile, log=False):
    '''
    parses the abstract xml file, returning list of papers maps
    '''
    list_results = []
    xml_doc = None 

    # parse the file
    with open(xmlfile) as xml_input:
        xml_doc = xmltodict.parse(xml_input.read())

    # loop through the papers
    for paper in xml_doc.get('PubmedArticleSet').get('PubmedArticle'):
        # empty paper dictionary
        map_paper = {}

        # get the pubmed id
        map_paper[COL_PUBMED] = paper.get('MedlineCitation').get('PMID').get('#text')

        # get the date
        if paper.get('MedlineCitation').get('DateRevised'):
            map_paper[COL_YEAR] = paper.get('MedlineCitation').get('DateRevised').get('Year')
            map_paper[COL_MONTH] = paper.get('MedlineCitation').get('DateRevised').get('Month')
            map_paper[COL_DAY] = paper.get('MedlineCitation').get('DateRevised').get('Day')

        # get the journal title
        map_paper[COL_JOURNAL] = paper.get('MedlineCitation').get('Article').get('Journal').get('ISOAbbreviation')

        # get the paper title
        map_paper[COL_TITLE] = paper.get('MedlineCitation').get('Article').get('ArticleTitle')

        # get the paper abstract 
        if paper.get('MedlineCitation').get('Article').get('Abstract'):
            map_paper[COL_ABSTRACT] = paper.get('MedlineCitation').get('Article').get('Abstract').get('AbstractText')


        # add to list
        list_results.append(map_paper)

    # return 
    return list_results

def db_delete_papers(conn, log=False):
    '''
    deletes the papers table
    '''
    # get the cursor
    cursor = conn.cursor()

    # delete
    cursor.execute(SQL_DELETE)

def db_insert_paper_list(conn, list_paper, log=False):
    '''
    inserts a list of papers into the corresponding DB tables
    '''
    num_count = 0

    # get the cursor
    cursor = conn.cursor()

    # loop through the papers
    for paper in list_paper:
        if paper.get(COL_ABSTRACT):
            num_count = num_count + 1
            paper[COL_DATE] = "{}-{}-{}".format(paper[COL_YEAR], paper[COL_MONTH], paper[COL_DAY])

            paper[COL_TITLE] = re.sub('<[A-Za-z\/][^>]*>', '', str(paper[COL_TITLE]))
            paper[COL_ABSTRACT] = re.sub('<[A-Za-z\/][^>]*>', '', str(paper[COL_ABSTRACT]))
            # print("{}\n".format(paper[COL_ABSTRACT]))
            # paper[COL_ABSTRACT] = BeautifulSoup(paper[COL_ABSTRACT]).getText()

            try:
                cursor.execute(SQL_INSERT, (paper[COL_PUBMED], paper[COL_TITLE], paper[COL_DATE], paper[COL_JOURNAL], paper[COL_ABSTRACT]))
            except mdb.err.DataError:
                print("skipping paper: {} with title {}".format(paper[COL_PUBMED], paper[COL_TITLE]))

            # commit 
            if num_count % 100 == 0:
                conn.commit()

    # final commit
    conn.commit()



# main
if __name__ == "__main__":
    # get the db connection
    conn = get_connection()

    # delete the papers table
    print("deleting data\n")
    db_delete_papers(conn)

    # get the input files
    files = [file for file in glob.glob(DIR_XML + "*.xml")]
    # files = [FILE_TEST_XML]
    for file_name in files:
        # get the papers for each file
        # list_papers = parse_abstract_file(FILE_TEST_XML)
        print("reading in file: {}".format(file_name))
        list_papers = parse_abstract_file(file_name)

        # insert the paper list
        print("inserting number records: {}\n".format(len(list_papers)))
        db_insert_paper_list(conn, list_papers)

        # # log
        # for row in list_papers:
        #     print("pubmed_id: {}".format(row))
