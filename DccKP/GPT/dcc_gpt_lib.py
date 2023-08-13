
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

SQL_SELECT_SEARCH = "select id from {}.pgpt_search where gene = %s".format(SCHEMA_GPT)
SQL_INSERT_SEARCH = "insert into {}.pgpt_search (name, terms, gene, to_download, to_download_ids) values(%s, %s, %s, %s, %s)".format(SCHEMA_GPT)
SQL_UPDATE_SEARCH = "update {}.pgpt_search set to_download_ids = %s, to_download = %s, ready = %s where id = %s".format(SCHEMA_GPT)
SQL_UPDATE_SEARCH_TO_DOWNLOAD = "update {}.pgpt_search set to_download = %s where id = %s".format(SCHEMA_GPT)
SQL_UPDATE_SEARCH_TO_DOWNLOAD_BY_GENE = "update {}.pgpt_search set to_download = %s where gene = %s".format(SCHEMA_GPT)



# methods
def get_connection(schema=SCHEMA_GPT):
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=schema)

    # return
    return conn


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


def file_to_string(file_path):
    ''' 
    read a file to a string
    '''
    with open(file_path, 'r') as f:
        return f.read()

