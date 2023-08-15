

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
from docx import Document
from docx.shared import Inches


# import relative libraries
dir_code = "/home/javaprog/Code/PythonWorkspace/"
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/GPT/')
import dcc_gpt_lib

# constants
DB_PASSWD = os.environ.get('DB_PASSWD')
SCHEMA_GPT = "gene_gpt"
DOC_FILENAME = "/home/javaprog/Data/Broad/GPT/GeneSummaries/geneSummary_{}_{}.docx"
SQL_SELECT_GENE_SUMMARIES = """
select se.gene, abst.abstract
from {}.pgpt_paper_abstract abst, {}.pgpt_search se 
where abst.search_top_level_of = se.id
order by se.gene;
""".format(SCHEMA_GPT, SCHEMA_GPT)

# methods
def get_db_gene_summaries(conn, list_genes=None, num_summaries=-1, log=False):
    '''
    get a list of abstract map objects
    '''
    # initialize
    list_summaries = []
    cursor = conn.cursor()

    # query 
    cursor.execute(SQL_SELECT_GENE_SUMMARIES)
    db_result = cursor.fetchall()
    for row in db_result:
        gene = row[0]
        abstract = row[1]
        if list_genes and gene not in list_genes:
            # skip if provided list of genes and not in it
            continue
        list_summaries.append({"gene": gene, 'summary': abstract})

    # return
    return list_summaries


def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=SCHEMA_GPT)

    # return
    return conn


# main
if __name__ == "__main__":
    # get the db connection
    conn = get_connection()
    list_genes_lipodystrophy = ['LMNA', 'PPARG', 'PLIN1', 'AGPAT2', 'BSCL2', 'CAV1', 'PTRF']
    list_genes_mody = ['GCK', 'HNF1A', 'HNF1B', 'CEL', 'PDX1', 'HNF4A', 'INS', 'NEUROD1', 'KLF11']
    list_genes = list_genes_mody

    # map
    map_gene_lists = {'T2D': dcc_gpt_lib.LIST_T2D, 'CAD': dcc_gpt_lib.LIST_CAD, 'Osteoarthritis': dcc_gpt_lib.LIST_OSTEO, 
                      'Kidney': dcc_gpt_lib.LIST_KCD, 'T1D': dcc_gpt_lib.LIST_T1D, 'Obesity': dcc_gpt_lib.LIST_OBESITY}

    # loop through map
    for key, value in map_gene_lists.items():
        # get the results
        list_genes = value
        # list_summaries = get_db_gene_summaries(conn=conn)
        list_summaries = get_db_gene_summaries(conn=conn, list_genes=list_genes)

        # create the document
        document = Document()

        # loop
        for item in list_summaries:
            # get the data
            gene = item.get('gene')
            summary = item.get('summary')

            # add to document
            document.add_heading('Gene: {}'.format(gene), 0)

            document.add_paragraph(summary, style='Intense Quote')

            document.add_page_break()

        # save the document
        file_document = DOC_FILENAME.format(key, round(time.time() * 1000))
        print("saving list tio dicument: {}".format(file_document))
        document.save(file_document)



    #     document.add_heading('Document Title', 0)

    #     p = document.add_paragraph('A plain paragraph having some ')
    #     p.add_run('bold').bold = True
    #     p.add_run(' and some ')
    #     p.add_run('italic.').italic = True

    #     document.add_heading('Heading, level 1', level=1)
    #     document.add_paragraph('Intense quote', style='Intense Quote')

    #     document.add_paragraph(
    #         'first item in unordered list', style='List Bullet'
    #     )
    #     document.add_paragraph(
    #         'first item in ordered list', style='List Number'
    #     )

    #     document.add_picture('monty-truth.png', width=Inches(1.25))

    #     document.add_page_break()

    # document.save('demo.docx')

