

# imports
import openai
import os 
import pymysql as mdb
from time import gmtime, strftime
import time
import json

# for AWS
ENV_DIR_CODE = os.environ.get('DIR_CODE')
ENV_DIR_PUBMED = os.environ.get('DIR_PUBMED')

# import relative libraries
dir_code = "/home/javaprog/Code/PythonWorkspace/"
if ENV_DIR_CODE:
    dir_code = ENV_DIR_CODE
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/GPT/')
print("using code dir: {}".format(dir_code))
import dcc_gpt_lib


# constants
# KEY_CHATGPT = os.environ.get('CHAT_KEY')
# openai.api_key = KEY_CHATGPT
# MODEL_CHATGPT="gpt-3.5-turbo-0301"
# MODEL_PROMPT_SUMMARIZE = "summarize the following in 200 words: \n{}"
# LIMIT_GPT_CALLS_PER_LEVEL = 45

# SEARCH_ID=1
# GPT_PROMPT = "summarize the information related to {} from the information below:\n"

# prompt
PROMPT_MAIN = """
Based on the abstracts provided from the following PubMed articles, identify and summarize the common themes or findings. Here are the abstracts:
{}
Please summarize the key commonalities for the gene {} among these studies and cite their PubMed IDs in your response.
"""
PROMPT_SUB_PAPER = """
{}. Title: {}
   PubMed ID: {}
   Abstract: {}
"""
PROMPT_MAIN_SHORT = """
Based on the abstracts provided from the following PubMed articles, identify and summarize in 400 words or less the common themes or findings. Here are the abstracts:
{}
Please summarize in 400 words or less the key commonalities for the gene {} among these studies and cite their PubMed IDs in your response.
"""

# db constants
DB_PASSWD = os.environ.get('DB_PASSWD')
NUM_ABSTRACT_LIMIT = 25
SCHEMA_GPT = "gene_gpt"
DB_PAPER_TABLE = "pgpt_paper"
DB_PAPER_ABSTRACT = "pgpt_paper_abtract"

# methods
def get_abstracts_for_gene(conn, gene, num_abstracts=3, to_shuffle=False, log=True):
    '''
    gets the absttracts for the gene
    '''
    # intialize
    list_abstracts = []

    # get the abstracts
    list_abstracts = dcc_gpt_lib.get_pubmed_abstracts_for_gene(conn=conn, gene=gene, num_abstracts=num_abstracts, to_shuffle=to_shuffle)

    # log
    if log:
        for row in list_abstracts:
            print("got pubmed id: {} with title: {}".format(row.get('pubmed_id'), row.get('title')))

    # return
    return list_abstracts


def build_reference_prompt(list_abstracts, log=True):
    '''
    builds the reference prompt
    '''
    # initilize
    str_abstracts = ""
    str_prompt = ""

    # loop
    for index, value in enumerate(list_abstracts):
        str_abstracts = str_abstracts + PROMPT_SUB_PAPER.format(index+1, value.get('title'), value.get('pubmed_id'), value.get('abstract'))

    # build the main abstract
    str_prompt = PROMPT_MAIN_SHORT.format(str_abstracts, gene)

    # log
    if log:
        print("got chat prompt: \n{}\n".format(str_prompt))

    # return
    return str_prompt


# main
if __name__ == "__main__":
    # initiliaze
    gene = 'PPARG'
    gene = 'SLC30A8'

    # get the connection
    conn = dcc_gpt_lib.get_connection(schema=SCHEMA_GPT)

    # get the abstracts
    print("getting abstracts for gene: {}".format(gene))
    list_abstracts = get_abstracts_for_gene(conn=conn, gene=gene, num_abstracts=25, to_shuffle=False, log=True)

    # get the prompt
    print("\ngetting prompt for gene: {}".format(gene))
    prompt = build_reference_prompt(list_abstracts=list_abstracts, log=True)































    # # get the name and prompt of the run
    # _, name_run, prompt_run = dcc_gpt_lib.get_db_run_data(conn=conn, id_run=id_run)
    # print("got run: {} with prompt: \n'{}'\n".format(name_run, prompt_run))

    # # get the list of searches
    # # list_searches = dcc_gpt_lib.get_db_list_ready_searches(conn=conn, num_searches=100)
    # list_searches = dcc_gpt_lib.get_db_list_search_genes_still_to_run(conn=conn, id_gpt_run=id_run, min_pubmed_count=min_pubmed, max_pubmed_count=max_pubmed, max_number=max_searches, log=True)
    # print("got searches to process count: {}".format(len(list_searches)))

    # # loop
    # index = 0
    # for search in list_searches:
    #     index = index + 1
    #     id_search = search.get('id')
    #     id_top_level_abstract = -1
    #     gene = search.get('gene')
    #     pubmed_count = search.get('pubmed_count')

    #     # log
    #     print("\n{}/{} - processing search: {} for gene: {} and pubmed count: {} for run id: {} of name: {}\n".format(index, len(list_searches), id_search, gene, pubmed_count, id_run, name_run))
    #     # time.sleep(5)
    #     time.sleep(3)
        
    #     try:
    #         # not anticipating to ever have 20 levels
    #         # for num_level in range(20):
    #         for num_level in range(5):
    #             # assume this is the top of the pyramid level until we find 2+ abstracts at this level
    #             found_top_level = True

    #             # get all the abstracts for the document level and run    max_pubmed = 45

    #             list_abstracts = dcc_gpt_lib.get_list_abstracts(conn=conn, id_search=id_search, id_run=id_run, num_level=num_level, num_abstracts=max_per_level, log=True)

    #             # if only one abstract and already above first level, then that is the final one, then set to final abstract and break
    #             if len(list_abstracts) == 1 and num_level > 0:
    #                 id_top_level_abstract = list_abstracts[0].get('id')
    #                 dcc_gpt_lib.update_db_abstract_for_search_and_run(conn=conn, id_abstract=id_top_level_abstract, id_search=id_search, id_run=id_run)
    #                 print("\nset top level: {} for search: {}, run: {} with abstract: {}".format(num_level, id_search, id_run, id_top_level_abstract))
    #                 print("==============================================================")
    #                 break

    #             # if not abstracts, then already done for this run and break
    #             elif len(list_abstracts) == 0:
    #                 print("\n\n\nalready done with no abstracts at level: {} for search: {}, run: {}".format(num_level, id_search, id_run))
    #                 # TODO - no break if no abstracts; should process ones that failed before
    #                 # break

    #             # split the abstracts into lists of size wanted and process
    #             else:
    #                 if len(list_abstracts) == 1 and num_level == 0:
    #                     print("only 1 abstract at level 0 for search: {}, so just summarize".format(id_search))

    #                 for i in range(0, len(list_abstracts), num_abstracts_per_summary):
    #                     i_end = i + num_abstracts_per_summary
    #                     print("using abstracts indexed at start: {} and end: {}".format(i, i_end))
    #                     list_sub = list_abstracts[i : i_end] 

    #                     # for the sub list
    #                     str_abstracts = ""
    #                     for item in list_sub:
    #                         abstract = item.get('abstract')
    #                         word_count = len(abstract.split())
    #                         # print("using abstract with count: {} and content: \n{}".format(word_count, abstract))
    #                         print("using abstract with id: {} and count: {}".format(item.get('id'), word_count))
    #                         str_abstracts = str_abstracts + "\n" + abstract

    #                     # log
    #                     print("using abstract count: {} for gpt query for level: {} and search: {}\n".format(len(list_sub), num_level, id_search))

    #                     # build the prompt
    #                     str_prompt = prompt_run.format(gene, gene, str_abstracts)

    #                     # get the chat gpt response
    #                     str_chat = call_chatgpt(str_query=str_prompt, log=True)
    #                     # print("using GPT prompt: \n{}".format(str_prompt))
    #                     print("\ngot chat gpt string: {}\n".format(str_chat))

    #                     # insert results and links
    #                     dcc_gpt_lib.insert_gpt_results(conn=conn, id_search=id_search, num_level=num_level, list_abstracts=list_abstracts, 
    #                                                 gpt_abstract=str_chat, id_run=id_run, name_run=name_run, log=True)
    #                     # time.sleep(30)
    #                     # time.sleep(1)

    #     except openai.error.Timeout:
    #         print("\n{}/{} Timeout ERROR ++++++++++++++ - skipping gene: {} with pubmed_count: {}".format(index, len(list_searches), gene, pubmed_count))
    #         time.sleep(120)
    #     except openai.error.APIError:
    #         print("\n{}/{} API ERROR ++++++++++++++ - skipping gene: {} with pubmed_count: {}".format(index, len(list_searches), gene, pubmed_count))
    #         time.sleep(120)
    #     except openai.error.ServiceUnavailableError:
    #         print("\n{}/{} Service unavailable ERROR ++++++++++++++ - skipping gene: {} with pubmed_count: {}".format(index, len(list_searches), gene, pubmed_count))
    #         time.sleep(120)
    #     except openai.error.APIConnectionError:
    #         print("\n{}/{} API Connection ERROR ++++++++++++++ - skipping gene: {} with pubmed_count: {}".format(index, len(list_searches), gene, pubmed_count))
    #         time.sleep(120)
    #     except json.decoder.JSONDecodeError:
    #         print("\n{}/{} Json (bad response) ERROR ++++++++++++++ - skipping gene: {} with pubmed_count: {}".format(index, len(list_searches), gene, pubmed_count))
    #         time.sleep(120)
    #     except mdb.err.DataError:
    #         print("\n{}/{} Got mysql ERROR ++++++++++++++ - skipping gene: {} with pubmed_count: {}".format(index, len(list_searches), gene, pubmed_count))
    #         time.sleep(3)
    #     except Exception as e:    
    #         if e: 
    #             print(e)   
    #         print("\n{}/{} Generic ERROR ++++++++++++++ - skipping gene: {} with pubmed_count: {}".format(index, len(list_searches), gene, pubmed_count))
    #         time.sleep(120)

