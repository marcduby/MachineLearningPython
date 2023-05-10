
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
TABLE_LINK = "pmd_link_keyword_abstract"
TABLE_KEYWORD = "pmd_keyword"

COL_PUBMED = "pubmed_id"
COL_YEAR = "num_year"
COL_MONTH = "num_month"
COL_DAY = "num_day"
COL_TITLE = "paper_title"
COL_JOURNAL = "journal_title"
COL_ABSTRACT = "abstract_text"
COL_DATE = "paper_date"
COL_LINKS = "keyword_links"

SQL_INSERT_ABSTRACT = """
    insert into {}.{} (pubmed_id, paper_title, paper_date, journal_title, abstract_text)
    values(%s, %s, %s, %s, %s)""".format(SCHEMA_GPT, TABLE_ABSTRACT)
SQL_INSERT_KEYWORD = "insert into {}.{} (keyword) values(%s)".format(SCHEMA_GPT, TABLE_KEYWORD)
SQL_INSERT_LINK = "insert into {}.{} (abstract_id, keyword_id, offset) values(%s, %s, %s)".format(SCHEMA_GPT, TABLE_LINK)

SQL_SELECT_ABSTRACT = "select id from {}.{} where pubmed_id = %s".format(SCHEMA_GPT, TABLE_ABSTRACT)
SQL_SELECT_KEYWORD = "select id from {}.{} where keyword = %s".format(SCHEMA_GPT, TABLE_KEYWORD)

SQL_DELETE_ABSTRACTS = "delete from {}.{}".format(SCHEMA_GPT, TABLE_ABSTRACT)
SQL_DELETE_LINKS = "delete from {}.{}".format(SCHEMA_GPT, TABLE_LINK)

FILE_TEST_XML = "/home/javaprog/Data/Broad/GPT/Pubmed/pubmed23n1165.xml"
# TEXT_JOURNAL_SUBNAME = "genet"
LIST_JOURNAL_SUBNAME = ["drug", "chem"]

# FILE_TEST_XML = "/home/javaprog/Code/PythonWorkspace/MachineLearningPython/DccKP/GPT/Pubmed/test.xml"
# TEXT_JOURNAL_SUBNAME = "me"

DIR_XML = "/home/javaprog/Data/Broad/GPT/Pubmed/"


def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=SCHEMA_GPT)

    # return
    return conn


def parse_abstract_file(xmlfile, list_journal_substring, log=False):
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
        # print(paper)
        # break
        # empty paper dictionary
        map_paper = {}

        # get the journal title
        map_paper[COL_JOURNAL] = paper.get('MedlineCitation').get('Article').get('Journal').get('ISOAbbreviation')
        if map_paper[COL_JOURNAL] and any([x in map_paper[COL_JOURNAL].lower() for x in list_journal_substring]):

            # get the pubmed id
            map_paper[COL_PUBMED] = paper.get('MedlineCitation').get('PMID').get('#text')

            # get the date
            if paper.get('MedlineCitation').get('DateRevised'):
                map_paper[COL_YEAR] = paper.get('MedlineCitation').get('DateRevised').get('Year')
                map_paper[COL_MONTH] = paper.get('MedlineCitation').get('DateRevised').get('Month')
                map_paper[COL_DAY] = paper.get('MedlineCitation').get('DateRevised').get('Day')

            # get the paper title
            map_paper[COL_TITLE] = paper.get('MedlineCitation').get('Article').get('ArticleTitle')

            # get the paper abstract 
            if paper.get('MedlineCitation').get('Article').get('Abstract'):
                map_paper[COL_ABSTRACT] = paper.get('MedlineCitation').get('Article').get('Abstract').get('AbstractText')

            # get the keywords
            map_paper[COL_LINKS] = set()
            # print(paper.get('MedlineCitation').get('MeshHeadingList'))
            if paper.get('MedlineCitation').get('MeshHeadingList'):
                for item in paper.get('MedlineCitation').get('MeshHeadingList').get('MeshHeading'):
                    if not isinstance(item, str):
                        keyword = item.get('DescriptorName').get('#text')
                        map_paper[COL_LINKS].add(keyword)
                        # print("got descriptor: {}".format(keyword))

            if paper.get('MedlineCitation').get('KeywordList'):
                if paper.get('MedlineCitation').get('KeywordList').get('Keyword'):
                    for item in paper.get('MedlineCitation').get('KeywordList').get('Keyword'):
                        if not isinstance(item, str):
                            keyword = item.get('#text')
                            map_paper[COL_LINKS].add(keyword)
                            # print("got keyword: {}".format(keyword))

            # add to list
            list_results.append(map_paper)

        elif log:
            print("skipping journal: {}".format(map_paper[COL_JOURNAL]))

    # return 
    return list_results

def get_keyword_id(conn, keyword, log=False):
    '''
    find keyword PK or return None
    '''
    keyword_id = None
    cursor = conn.cursor()

    # find
    # print(keyword)
    # print(SQL_SELECT_KEYWORD)
    cursor.execute(SQL_SELECT_KEYWORD, (keyword))
    db_result = cursor.fetchall()
    if db_result:
        keyword_id = db_result[0][0]

    # return 
    return keyword_id

def get_or_insert_keyword_id(conn, keyword, log=False):
    '''
    will retrive the db PK for the keyword and insert if not there 
    '''
    keyword_id = None 

    # find the keyword 
    keyword_id = get_keyword_id(conn, keyword)

    # if not found, insert 
    if not keyword_id:
        cursor = conn.cursor()
        cursor.execute(SQL_INSERT_KEYWORD, (keyword))
        conn.commit()

    # get id now 
    keyword_id = get_keyword_id(conn, keyword)

    # return
    return keyword_id

def db_delete_papers(conn, log=False):
    '''
    deletes the papers table
    '''
    # get the cursor
    cursor = conn.cursor()

    # delete
    cursor.execute(SQL_DELETE_ABSTRACTS)
    cursor.execute(SQL_DELETE_LINKS)
    conn.commit()

def db_insert_paper_list(conn, list_paper, log=False):
    '''
    inserts a list of papers into the corresponding DB tables
    '''
    num_count = 0

    # get the cursor
    cursor = conn.cursor()

    # loop through the papers
    for paper in list_paper:
        # see if paper is already in the db
        pubmed_id = paper.get(COL_PUBMED)
        if is_abstract_in_db(cursor, pubmed_id):
            print("skipping paper: {} that is already in db".format(pubmed_id))
            continue
        
        if paper.get(COL_ABSTRACT):
            num_count = num_count + 1
            paper[COL_DATE] = "{}-{}-{}".format(paper[COL_YEAR], paper[COL_MONTH], paper[COL_DAY])

            paper[COL_TITLE] = re.sub('<[A-Za-z\/][^>]*>', '', str(paper[COL_TITLE]))
            paper[COL_ABSTRACT] = re.sub('<[A-Za-z\/][^>]*>', '', str(paper[COL_ABSTRACT]))
            # print("{}\n".format(paper[COL_ABSTRACT]))
            # paper[COL_ABSTRACT] = BeautifulSoup(paper[COL_ABSTRACT]).getText()

            try:
                cursor.execute(SQL_INSERT_ABSTRACT, (paper[COL_PUBMED], paper[COL_TITLE], paper[COL_DATE], paper[COL_JOURNAL], paper[COL_ABSTRACT]))
                print("\ninserted pubmed: {} from journal: {}".format(paper[COL_PUBMED], paper[COL_JOURNAL]))

                # insert the keyword links
                db_insert_abstract_links(cursor, paper[COL_PUBMED], paper[COL_ABSTRACT], paper[COL_LINKS], log=log)

            except mdb.err.DataError:
                print("skipping paper: {} with title {}".format(paper[COL_PUBMED], paper[COL_TITLE]))

            # commit 
            if num_count % 10 == 0:
                conn.commit()

    # final commit
    conn.commit()

def db_insert_abstract_links(cursor, pubmed_id, text_abstract, set_links, log=False):
    '''
    insert the abstract keyword links 
    '''
    # cursor = conn.cursor()
    abstract_id = None 

    # get the abstract id
    cursor.execute(SQL_SELECT_ABSTRACT, (pubmed_id))
    db_result = cursor.fetchall()
    if db_result and len(db_result) == 1:
        abstract_id = db_result[0][0]

    if abstract_id:
        # for each keyword
        for item in set_links:
            if item:
                # get the keyword position in the abstract
                offset = text_abstract.lower().find(item.lower())
                if offset >= 0:
                    # get the keyword id
                    keyword_id = get_or_insert_keyword_id(conn, item)
                    print("found in abstract: {} for pubmed: {}".format(item, pubmed_id))
                    if keyword_id:
                        # insert
                        cursor.execute(SQL_INSERT_LINK, (abstract_id, keyword_id, offset))
                elif True:
                    if item.lower().find('sotos') >= 0:
                        print("not found in abstract, skipping: {} for pubmed: {} - {}".format(item, pubmed_id, text_abstract))
    else:
        print("found no abstract id for pubmed: {}".format(pubmed_id))


def is_abstract_in_db(cursor, pubmed_id, log=False):
    '''
    will return boolean if pubmed id already in db
    '''
    is_present = False

    # search the db
    cursor.execute(SQL_SELECT_ABSTRACT, (pubmed_id))
    db_result = cursor.fetchall()
    if db_result:
        is_present = True

    # return
    return is_present


# main
if __name__ == "__main__":
    # get the db connection
    conn = get_connection()

    # delete the papers table
    # print("deleting data\n")
    # db_delete_papers(conn)

    # get the input files
    files = [file for file in glob.glob(DIR_XML + "*.xml")]
    # files = [FILE_TEST_XML]
    for file_name in files:
        # get the papers for each file
        # list_papers = parse_abstract_file(FILE_TEST_XML)
        print("reading in file: {}, only keep journals with sub name: {}".format(file_name, LIST_JOURNAL_SUBNAME))
        list_papers = parse_abstract_file(file_name, LIST_JOURNAL_SUBNAME)

        # insert the paper list
        print("inserting number records: {}\n".format(len(list_papers)))
        db_insert_paper_list(conn, list_papers, log=False)

        # # log
        # for row in list_papers:
        #     print("pubmed_id: {}".format(row))
