
# imports
import os
import pymysql as mdb
import xmltodict
import json

# constants
DB_PASSWD = os.environ.get('DB_PASSWD')


# gene lists
LIST_T2D = "PAM,TBC1D4,WFS1,ANGPTL4,GCK,GLIS3,GLP1R,KCNJ11,MC4R,PDX1"
LIST_KCD = "ALKAL2,BCL2L14,BIN3,F12,FAM102A,FAM53B,FGF5,IRF5,KNG1,MFSD4A,OVOL1,SESN2,SHROOM3,SLC47A1,TMEM125,TTYH3,TUB,UBASH3B,UBE2D3,UMOD,USP2"
LIST_OSTEO = "CHST3,FBN2,FGFR3,GDF5,LMX1B,LTBP3,MGP,SMAD3,WNT10B"
LIST_CAD = "PCSK9,NOS3,BSND,LPL,LIPA,ANGPTL4,LDLR"
LIST_T1D = "CTLA4,CTSH,C1QTNF6,INS,IL2RA,AIRE,CCR5,GLIS3,IFIH1,PRF1,PTPN22,SH2B3,TYK2,CEL,FUT2,MICA,ZFP57,SIRPG"
LIST_OBESITY = "FTO,MC4R,HSD17B12,INO80E,MAP2K5,NFATC2IP,POC5,SH2B1,TUFM"

LIST_MASTER = [LIST_T2D, LIST_KCD, LIST_OSTEO, LIST_CAD, LIST_T1D, LIST_OBESITY]

# schema
SCHEMA_GPT = "gene_gpt"

# took out schema reference since will depned on connection
SQL_SELECT_SEARCH = "select id from pgpt_search where gene = %s"
SQL_INSERT_SEARCH = "insert into pgpt_search (name, terms, gene, to_download, to_download_ids) values(%s, %s, %s, %s, %s)"
SQL_UPDATE_SEARCH = "update pgpt_search set to_download_ids = %s, to_download = %s, ready = %s where id = %s"
SQL_UPDATE_SEARCH_TO_DOWNLOAD = "update pgpt_search set to_download = %s where id = %s"
SQL_UPDATE_SEARCH_TO_DOWNLOAD_BY_GENE = "update pgpt_search set to_download = %s where gene = %s"
SQL_UPDATE_SEARCH_READY_BY_GENE = "update pgpt_search set ready = %s where gene = %s"

SQL_INSERT_ABSTRACT = "insert into pgpt_paper_abstract (pubmed_id, abstract, title, journal_name, paper_year, document_level, in_pubmed_file) values(%s, %s, %s, %s, %s, %s, %s)"
SQL_SELECT_ABSTRACT_IF_NOT_DOWNLOADED = """
SELECT paper.pubmed_id 
FROM pgpt_paper paper LEFT JOIN pgpt_paper_abstract abstract
ON paper.pubmed_id = abstract.pubmed_id WHERE abstract.id IS NULL;
"""

SQL_SELECT_PAPER_ALL = "select pubmed_id from pgpt_paper"
SQL_SELECT_ABSTRACT_BY_PUBMED_ID = "select id from pgpt_paper_abstract where pubmed_id = %s"

SQL_SELECT_ABSTRACT_FILES = "select distinct in_pubmed_file from pgpt_paper_abstract"

SQL_SELECT_FILE_FOR_RUN = "select file_name from pgpt_file_run where run_name = %s and is_done = 'Y'"
SQL_INSERT_FILE_RUN = "insert into pgpt_file_run (file_name, run_name, is_done) values(%s, %s, %s)"

SQL_INSERT_PUBMED_REFERENCE = "insert ignore into pgpt_paper_reference (pubmed_id, referring_pubmed_id) select pap.pubmed_id, %s from pgpt_paper pap where pap.pubmed_id = %s"

# SQL_INSERT_SEARCH = "insert into {}.pgpt_search (name, terms, gene, to_download, to_download_ids) values(%s, %s, %s, %s, %s)".format(SCHEMA_GPT)
# SQL_UPDATE_SEARCH = "update {}.pgpt_search set to_download_ids = %s, to_download = %s, ready = %s where id = %s".format(SCHEMA_GPT)
# SQL_UPDATE_SEARCH_TO_DOWNLOAD = "update {}.pgpt_search set to_download = %s where id = %s".format(SCHEMA_GPT)
# SQL_UPDATE_SEARCH_TO_DOWNLOAD_BY_GENE = "update {}.pgpt_search set to_download = %s where gene = %s".format(SCHEMA_GPT)

# SQL_INSERT_ABSTRACT = "insert into {}.pgpt_paper_abstract (pubmed_id, abstract, title, journal_name, paper_year, document_level, in_pubmed_file) values(%s, %s, %s, %s, %s, %s, %s)".format(SCHEMA_GPT)
# SQL_SELECT_ABSTRACT_IF_DOWNLOADED = """
# SELECT paper.pubmed_id 
# FROM {}.pgpt_paper paper LEFT JOIN {}.pgpt_paper_abstract abstract
# ON paper.pubmed_id = abstract.pubmed_id WHERE abstract.id IS NULL and paper.pubmed_id = %s;
# """.format(SCHEMA_GPT, SCHEMA_GPT)

# SQL_SELECT_PAPER_ALL = "select pubmed_id from {}.pgpt_paper".format(SCHEMA_GPT)

# methods
def get_connection(schema=SCHEMA_GPT):
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=schema)

    # return
    return conn

def insert_db_pubmed_reference(conn, pubmed_id, ref_pubmed_id, is_commit='N', log=False):
    '''
    inserts the pubmed reference
    '''
    # initialize
    cursor = conn.cursor()

    # find
    cursor.execute(SQL_INSERT_PUBMED_REFERENCE, (ref_pubmed_id, pubmed_id))
    if is_commit:
        conn.commit()

def get_db_completed_file_runs(conn, run_name, log=False):
    '''
    will return all the file names of the references already processed
    '''
    # initialize
    list_result = []
    cursor = conn.cursor()

    # pick query 
    sql_select = SQL_SELECT_FILE_FOR_RUN

    # find
    cursor.execute(sql_select, (run_name))
    db_result = cursor.fetchall()
    for row in db_result:
        if row[0]:
           list_result.append(row[0])

    return list_result

def insert_db_file_run(conn, file_name, run_name, completed='Y', log=False):
    '''
    inserts the file and run into the log table
    '''
    # initialize
    cursor = conn.cursor()

    # find
    cursor.execute(SQL_INSERT_FILE_RUN, (file_name, run_name, completed))
    conn.commit()

def get_db_abstract_files_processed(conn, log=False):
    '''
    will return all the file names of the abstract files already processed
    '''
    # initialize
    list_result = []
    cursor = conn.cursor()

    # pick query 
    sql_select = SQL_SELECT_ABSTRACT_FILES

    # find
    cursor.execute(sql_select, ())
    db_result = cursor.fetchall()
    for row in db_result:
        if row[0]:
           list_result.append(row[0])

    # return 
    return set(list_result)

def get_db_all_pubmed_ids(conn, log=False):
    '''
    will return all the pubmed ids in the database
    '''
    # initialize
    list_result = []
    cursor = conn.cursor()

    # pick query 
    sql_select = SQL_SELECT_ABSTRACT_IF_NOT_DOWNLOADED

    # find
    cursor.execute(sql_select, ())
    db_result = cursor.fetchall()
    for row in db_result:
        list_result.append(int(row[0]))

    # return 
    return set(list_result)

def get_db_if_pubmed_downloaded(conn, pubmed_id, log=False):
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

def insert_db_paper_abstract(conn, pubmed_id, abstract, title, journal, year, file_name=None, document_level=0, log=False):
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
        if abstract and journal and title:
            cursor.execute(SQL_INSERT_ABSTRACT, (pubmed_id, abstract, title, journal, year, document_level, file_name))
            conn.commit()
        else:
            print("GOT ERROR: skipping empty abstract for pubmed id: {}".format(pubmed_id))
    except mdb.err.DataError:
        print("GOT DATABASE ERROR: skipping empty abstract for pubmed id: {}".format(pubmed_id))


def get_db_search_by_gene(conn, gene, log=False):
    '''
    find search id for a gene
    '''
    result_id = None
    cursor = conn.cursor()

    # pick query 
    sql_select = SQL_SELECT_SEARCH

    # find
    cursor.execute(sql_select, (gene))
    db_result = cursor.fetchall()
    if db_result:
        result_id = db_result[0][0]

    # return 
    return result_id

def insert_db_search(conn, gene, to_dowwnload = 'N', to_download_ids = 'Y', log=False):
    '''
    will insert a gene search
    '''
    # initialize
    result_id = None
    cursor = conn.cursor()

    # see if already in db
    result_id = get_db_search_by_gene(conn=conn, gene=gene, log=log)

    # if not, insert
    if not result_id:
        if log:
            print("inserting search for gene: {}".format(gene))
        cursor.execute(SQL_INSERT_SEARCH, (gene + " search", gene + ",human", gene, to_dowwnload, to_download_ids))
        conn.commit()

def update_db_search(conn, id_search, ready = 'N', to_download = 'N', to_download_ids = 'Y', log=False):
    '''
    will update a gene search
    '''
    # initialize
    cursor = conn.cursor()

    # update
    cursor.execute(SQL_UPDATE_SEARCH, (to_download_ids, to_download, ready, id_search))
    conn.commit()

def update_db_search_to_download(conn, id_search, to_download = 'N', log=False):
    '''
    will update a gene search
    '''
    # initialize
    cursor = conn.cursor()

    # update
    cursor.execute(SQL_UPDATE_SEARCH_TO_DOWNLOAD, (to_download, id_search))
    conn.commit()

def update_db_search_ready_by_gene(conn, gene, ready = 'N', log=False):
    '''
    will update a gene search ready flag
    '''
    # initialize
    cursor = conn.cursor()

    # update
    cursor.execute(SQL_UPDATE_SEARCH_READY_BY_GENE, (ready, gene))
    conn.commit()

def update_db_search_to_download_by_gene(conn, gene, to_download = 'N', log=False):
    '''
    will update a gene search
    '''
    # initialize
    cursor = conn.cursor()

    # update
    cursor.execute(SQL_UPDATE_SEARCH_TO_DOWNLOAD_BY_GENE, (to_download, gene))
    conn.commit()

def get_pubmed_article_list(xml_input, log=False):
    '''
    get the pubmed article list from the given xml text input
    '''
    # initialize
    list_result = None

    # convert 
    try:
        map_response = xmltodict.parse(xml_input)

        if log:
            print("got rest xml input: {}".format(json.dumps(map_response, indent=1)))

        if map_response.get('PubmedArticleSet'):
            temp = map_response.get('PubmedArticleSet').get('PubmedArticle')
            if temp:
                if isinstance(temp, list):
                    list_result = temp
                else:
                    list_result = [temp]

    except xml.parsers.expat.ExpatError:
        print("GOT ERROR: {}".format(xml_input))

    # return
    return list_result

def get_paper_data_from_map(map_paper, log=False):
    '''
    extracts the abstract, journal and other data from the given map data
    '''
    # initialize
    list_temp = []
    text_abstract = None
    title = ""
    journal = ""
    year = 0
    id_pubmed = None

    # get the data
    resp_article = map_paper.get('MedlineCitation').get('Article')
    if resp_article and resp_article.get('Abstract'):
        list_abstract = resp_article.get('Abstract').get('AbstractText')
        if log:
            print(list_abstract)

        if list_abstract:
            if isinstance(list_abstract, list):
                for item in list_abstract:
                    if item:
                        if isinstance(item, str):
                            list_temp.append(item)
                        else:
                            if item.get('#text'):
                                list_temp.append(item.get('#text'))
                text_abstract = " ".join(list_temp)
            elif isinstance(list_abstract, dict):
                text_abstract = list_abstract.get('#text')
            elif isinstance(list_abstract, str):
                text_abstract = list_abstract

    # FIX: move this out of abstract test since none abstract still has pubmed id and other data
    # get year
    if map_paper.get('MedlineCitation').get('DateCompleted'):
        year = map_paper.get('MedlineCitation').get('DateCompleted').get('Year')
    title = map_paper.get('MedlineCitation').get('Article').get('ArticleTitle')
    if isinstance(title, dict):
        title = title.get('#text')
    journal = map_paper.get('MedlineCitation').get('Article').get('Journal').get('Title')
    id_pubmed = map_paper.get('MedlineCitation').get('PMID').get('#text')
    if id_pubmed:
        id_pubmed = int(id_pubmed)

    # return
    return text_abstract, title, journal, year, id_pubmed

def get_paper_references_from_map(map_paper, log=False):
    '''
    extracts the abstract, journal and other data from the given map data
    '''
    # initialize
    id_pubmed = None
    list_temp = []

    data_pubmed = map_paper.get('PubmedData')
    if data_pubmed:
        map_article = data_pubmed.get('ArticleIdList')

        # get the pubmed id
        list_article = map_article.get('ArticleId')
        id_pubmed = get_pubmed_id_from_article_id_list(list_article=list_article)

        if id_pubmed:
            id_pubmed = int(id_pubmed)
            # find the references
            if data_pubmed.get('ReferenceList'):
                if not isinstance(data_pubmed.get('ReferenceList'), list):
                    list_reference = data_pubmed.get('ReferenceList').get('Reference')
                    if log:
                        print(json.dumps(list_reference, indent=1))
                    if list_reference:
                        if isinstance(list_reference, list):
                            for item in list_reference:
                                id_reference = get_pubmed_reference_from_map(item)
                                if id_reference:
                                    list_temp.append(int(id_reference))

                            #     if item.get('ArticleIdList'):
                            #         if item.get('ArticleIdList').get('ArticleId'):
                            #             id_reference = get_pubmed_id_from_article_id_list(item.get('ArticleIdList').get('ArticleId'))
                            #             if id_reference:
                            #                 list_temp.append(int(id_reference))
                        else:
                            id_reference = get_pubmed_reference_from_map(list_reference)
                            if id_reference:
                                list_temp.append(int(id_reference))
                else:                            
                    print("GOT ODD REFRENCE: \n{}".format(json.dumps(data_pubmed.get('lis'), indent=1)))

    # return
    return id_pubmed, list_temp

def get_pubmed_reference_from_map(item, log=False):
    '''
    extracts the reference list from the map
    '''
    id_reference = None

    if item.get('ArticleIdList'):
        if item.get('ArticleIdList').get('ArticleId'):
            id_reference = get_pubmed_id_from_article_id_list(item.get('ArticleIdList').get('ArticleId'))

    return id_reference

def get_pubmed_id_from_article_id_list(list_article, log=False):
    '''
    extract the ids from a map of article llist
    '''
    id_pubmed = None 
    map_id = None

    # extract the data
    if list_article:
        if isinstance(list_article, list):
            for item in list_article:
                if item.get('@IdType') and item.get('@IdType') == 'pubmed':
                    map_id = item
        else:
            if list_article.get('@IdType') and list_article.get('@IdType') == 'pubmed':
                map_id = list_article

    if map_id:
        id_pubmed = int(map_id.get('#text'))

    # return
    return id_pubmed


def file_to_string(file_path):
    ''' 
    read a file to a string
    '''
    with open(file_path, 'r') as f:
        return f.read()

def get_all_files_in_directory(dir_input, log=False):
    '''
    get all the files in a directory
    '''
    list_files = []

    # get all the files
    # Iterate over directory contents
    for entry in os.scandir(dir_input):
        # If entry is a file, store it
        if entry.is_file():
            list_files.append(entry.name) 

    # return
    return list_files

