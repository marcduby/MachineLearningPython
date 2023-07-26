

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


# constants
DB_PASSWD = os.environ.get('DB_PASSWD')
SCHEMA_GPT = "gene_gpt"
DOC_FILENAME = "/home/javaprog/Data/Broad/GPT/GeneSummaries/geneSummary_{}.docx"
SQL_SELECT_GENE_SUMMARIES = """
select se.gene, abst.abstract
from {}.pgpt_paper_abstract abst, {}.pgpt_search se 
where abst.search_top_level_of = se.id
order by se.gene;
""".format(SCHEMA_GPT, SCHEMA_GPT)

# methods
def get_db_gene_summaries(conn, num_summaries=-1, log=False):
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

    # get the results
    list_summaries = get_db_gene_summaries(conn=conn)

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
    document.save(DOC_FILENAME.format(round(time.time() * 1000)))



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

