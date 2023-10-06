

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
import dcc_gpt_lib

# constants
STR_INPUT1 = "We performed collapsing analyses on 454,796 UK Biobank (UKB) exomes to detect gene-level associations with diabetes. Recessive carriers of nonsynonymous variants in  were 30% less likely to develop diabetes ( = 5.7 × 10) and had lower glycosylated hemoglobin (β = -0.14 SD units,  = 1.1 × 10). These associations were independent of body mass index, suggesting protection against insulin resistance even in the setting of obesity. We replicated these findings in 96,811 Admixed Americans in the Mexico City Prospective Study ( < 0.05)Moreover, the protective effect of  variants was stronger in individuals who did not carry the Latino-enriched  risk haplotype ( = 6.0 × 10). Separately, we identified a Finnish-enriched  protein-truncating variant associated with decreased odds of both type 1 and type 2 diabetes ( < 0.05) in FinnGen. No adverse phenotypes were associated with protein-truncating  variants in the UKB, supporting this gene as a therapeutic target for diabetes."
STR_INPUT2 = "A major goal in human genetics is to use natural variation to understand the phenotypic consequences of altering each protein-coding gene in the genome. Here we used exome sequencing to explore protein-altering variants and their consequences in 454,787 participants in the UK Biobank study. We identified 12 million coding variants, including around 1 million loss-of-function and around 1.8 million deleterious missense variants. When these were tested for association with 3,994 health-related traits, we found 564 genes with trait associations at P ≤ 2.18 × 10. Rare variant associations were enriched in loci from genome-wide association studies (GWAS), but most (91%) were independent of common variant signals. We discovered several risk-increasing associations with traits related to liver disease, eye disease and cancer, among others, as well as risk-lowering associations for hypertension (SLC9A3R2), diabetes (MAP3K15, FAM234A) and asthma (SLC27A3). Six genes were associated with brain imaging phenotypes, including two involved in neural development (GBE1, PLD1). Of the signals available and powered for replication in an independent cohort, 81% were confirmed; furthermore, association signals were generally consistent across individuals of European, Asian and African ancestry. We illustrate the ability of exome sequencing to identify gene-trait associations, elucidate gene function and pinpoint effector genes that underlie GWAS signals at scale."
STR_INPUT3 = "Mitogen-activated protein kinases (MAP kinases) are functionally connected kinases that regulate key cellular process involved in kidney disease such as all survival, death, differentiation and proliferation. The typical MAP kinase module is composed by a cascade of three kinases: a MAP kinase kinase kinase (MAP3K) that phosphorylates and activates a MAP kinase kinase (MAP2K) which phosphorylates a MAP kinase (MAPK). While the role of MAPKs such as ERK, p38 and JNK has been well characterized in experimental kidney injury, much less is known about the apical kinases in the cascade, the MAP3Ks. There are 24 characterized MAP3K (MAP3K1 to MAP3K21 plus RAF1, BRAF and ARAF). We now review current knowledge on the involvement of MAP3K in non-malignant kidney disease and the therapeutic tools available. There is in vivo interventional evidence clearly supporting a role for MAP3K5 (ASK1) and MAP3K14 (NIK) in the pathogenesis of experimental kidney disease. Indeed, the ASK1 inhibitor Selonsertib has undergone clinical trials for diabetic kidney disease. Additionally, although MAP3K7 (MEKK7, TAK1) is required for kidney development, acutely targeting MAP3K7 protected from acute and chronic kidney injury; and targeting MAP3K8 (TPL2/Cot) protected from acute kidney injury. By contrast MAP3K15 (ASK3) may protect from hypertension and BRAF inhibitors in clinical use may induced acute kidney injury and nephrotic syndrome. Given their role as upstream regulators of intracellular signaling, MAP3K are potential therapeutic targets in kidney injury, as demonstrated for some of them. However, the role of most MAP3K in kidney disease remains unexplored."
KEY_CHATGPT = os.environ.get('CHAT_KEY')
openai.api_key = KEY_CHATGPT
MODEL_CHATGPT="gpt-3.5-turbo-0301"
MODEL_PROMPT_SUMMARIZE = "summarize the following in 200 words: \n{}"
LIMIT_GPT_CALLS_PER_LEVEL = 45

SEARCH_ID=1
GPT_PROMPT = "summarize the information related to {} from the information below:\n"

# db constants
DB_PASSWD = os.environ.get('DB_PASSWD')
NUM_ABSTRACT_LIMIT = 5
SCHEMA_GPT = "gene_gpt"
DB_PAPER_TABLE = "pgpt_paper"
DB_PAPER_ABSTRACT = "pgpt_paper_abtract"

# SQL_SELECT_ABSTRACT_BY_TITLE = "select id from {}.pgpt_paper_abstract where title = %s".format(SCHEMA_GPT)
# SQL_SELECT_ABSTRACT_LIST_LEVEL_0 = """
# select abst.id, abst.abstract 
# from {}.pgpt_paper_abstract abst, {}.pgpt_search_paper seapaper 
# where abst.document_level = 0 and seapaper.pubmed_id = abst.pubmed_id and seapaper.search_id = %s
# and abst.id not in (select child_id from {}.pgpt_gpt_paper where search_id = %s) limit %s
# """.format(SCHEMA_GPT, SCHEMA_GPT, SCHEMA_GPT)

# SQL_SELECT_ABSTRACT_LIST_LEVEL_HIGHER = """
# select distinct abst.id, abst.abstract, abst.document_level
# from {}.pgpt_paper_abstract abst, {}.pgpt_gpt_paper gpt
# where abst.document_level = %s and gpt.parent_id = abst.id and gpt.search_id = %s
# and abst.id not in (select child_id from {}.pgpt_gpt_paper where search_id = %s) limit %s
# """.format(SCHEMA_GPT, SCHEMA_GPT, SCHEMA_GPT)

# SQL_INSERT_PAPER = "insert into {}.pgpt_paper (pubmed_id) values(%s)".format(SCHEMA_GPT)
# SQL_INSERT_ABSTRACT = "insert into {}.pgpt_paper_abstract (abstract, title, journal_name, document_level) values(%s, %s, %s, %s)".format(SCHEMA_GPT)
# SQL_INSERT_GPT_LINK = "insert into {}.pgpt_gpt_paper (search_id, parent_id, child_id, document_level) values(%s, %s, %s, %s)".format(SCHEMA_GPT)
# SQL_UPDATE_ABSTRACT_FOR_TOP_LEVEL = "update {}.pgpt_paper_abstract set search_top_level_of = %s where id = %s".format(SCHEMA_GPT)

SQL_UPDATE_SEARCH_AFTER_SUMMARY = "update {}.pgpt_search set ready = 'N', date_last_summary = sysdate() where id = %s ".format(SCHEMA_GPT)

# counts each paper multiple times due to not distinct
SQL_SEARCH_COUNT_FOR_LEVEL = "select count(sp.id) from pgpt_gpt_paper sp, pgpt_search se where sp.search_id = se.id and se.id = %s and sp.document_level = %s".format(SCHEMA_GPT)

# methods
def call_chatgpt(str_query, log=False):
    '''
    makes the api call to chat gpt service
    '''
    # initialize
    str_result = ""
    list_conversation = []

    # build the payload
    # list_conversation.append({'role': 'system', 'content': MODEL_PROMPT_SUMMARIZE.format(str_query)})
    list_conversation.append({'role': 'system', 'content': str_query})

    if log:
        print("using chat input: {}\n".format(json.dumps(list_conversation, indent=1)))

    # query
    response = openai.ChatCompletion.create(
        model = MODEL_CHATGPT,
        messages = list_conversation
    )

    # get the response
    str_response = response.choices
    # log
    if log:
        print("got chatGPT response: {}".format(str_response))

    # get the text response
    str_result = str_response[0].get('message').get('content')

    # return
    return str_result

# main
if __name__ == "__main__":
    # initiliaze
    # # list_input = [STR_INPUT1, STR_INPUT2, STR_INPUT3]
    # # str_input = " ".join(list_input)    
    # num_level = 1
    # id_search = 1
    num_abstracts_per_summary = 5
    gpt_prompt = GPT_PROMPT.format("PPARG")
    id_run = 8
    # max_per_level = 50
    # max_pubmed = 25
    max_per_level = 25
    max_pubmed = 35
    min_pubmed = 2
    max_searches = 5000

    # # get the chat gpt response
    # str_chat = call_chatgpt(str_input, log=True)
    # print("got chat gpt string: {}".format(str_chat))


    # get the connection
    conn = dcc_gpt_lib.get_connection()

    # get the name and prompt of the run
    _, name_run, prompt_run = dcc_gpt_lib.get_db_run_data(conn=conn, id_run=id_run)
    print("got run: {} with prompt: \n'{}'\n".format(name_run, prompt_run))

    # get the list of searches
    # list_searches = dcc_gpt_lib.get_db_list_ready_searches(conn=conn, num_searches=100)
    list_searches = dcc_gpt_lib.get_db_list_search_genes_still_to_run(conn=conn, id_gpt_run=id_run, min_pubmed_count=min_pubmed, max_pubmed_count=max_pubmed, max_number=max_searches, log=True)
    print("got searches to process count: {}".format(len(list_searches)))

    # loop
    index = 0
    for search in list_searches:
        index = index + 1
        id_search = search.get('id')
        id_top_level_abstract = -1
        gene = search.get('gene')
        pubmed_count = search.get('pubmed_count')

        # log
        print("\n{}/{} - processing search: {} for gene: {} and pubmed count: {} for run id: {} of name: {}\n".format(index, len(list_searches), id_search, gene, pubmed_count, id_run, name_run))
        # time.sleep(5)
        time.sleep(3)
        
        try:
            # not anticipating to ever have 20 levels
            for num_level in range(20):
                # assume this is the top of the pyramid level until we find 2+ abstracts at this level
                found_top_level = True

                # get all the abstracts for the document level and run
                list_abstracts = dcc_gpt_lib.get_list_abstracts(conn=conn, id_search=id_search, id_run=id_run, num_level=num_level, num_abstracts=max_per_level, log=True)

                # if only one abstract, then set to final abstract and break
                if len(list_abstracts) == 1:
                    if num_level > 0:
                        id_top_level_abstract = list_abstracts[0].get('id')
                        dcc_gpt_lib.update_db_abstract_for_search_and_run(conn=conn, id_abstract=id_top_level_abstract, id_search=id_search, id_run=id_run)
                        print("\nset top level: {} for search: {}, run: {} with abstract: {}".format(num_level, id_search, id_run, id_top_level_abstract))
                    else:
                        print("only 1 abstract at level 0 for ")
                    print("==============================================================")
                    break

                # if not abstracts, then already done for this run and break
                elif len(list_abstracts) == 0:
                    print("\n\n\nalready done with no abstracts at level: {} for search: {}, run: {}".format(num_level, id_search, id_run))
                    break

                # split the abstracts into lists of size wanted and process
                else:
                    for i in range(0, len(list_abstracts), num_abstracts_per_summary):
                        i_end = i + num_abstracts_per_summary
                        print("using abstracts indexed at start: {} and end: {}".format(i, i_end))
                        list_sub = list_abstracts[i : i_end] 

                        # for the sub list
                        str_abstracts = ""
                        for item in list_sub:
                            abstract = item.get('abstract')
                            word_count = len(abstract.split())
                            # print("using abstract with count: {} and content: \n{}".format(word_count, abstract))
                            print("using abstract with id: {} and count: {}".format(item.get('id'), word_count))
                            str_abstracts = str_abstracts + "\n" + abstract

                        # log
                        print("using abstract count: {} for gpt query for level: {} and search: {}\n".format(len(list_sub), num_level, id_search))

                        # build the prompt
                        str_prompt = prompt_run.format(gene, gene, str_abstracts)

                        # get the chat gpt response
                        str_chat = call_chatgpt(str_query=str_prompt, log=True)
                        # print("using GPT prompt: \n{}".format(str_prompt))
                        print("\ngot chat gpt string: {}\n".format(str_chat))

                        # insert results and links
                        dcc_gpt_lib.insert_gpt_results(conn=conn, id_search=id_search, num_level=num_level, list_abstracts=list_abstracts, 
                                                    gpt_abstract=str_chat, id_run=id_run, name_run=name_run, log=True)
                        # time.sleep(30)
                        # time.sleep(1)

        except openai.error.Timeout:
            print("\n{}/{} Timeout ERROR ++++++++++++++ - skipping gene: {} with pubmed_count: {}".format(index, len*list_searches, gene, pubmed_count))
            time.sleep(120)
        except openai.error.APIError:
            print("\n{}/{} API ERROR ++++++++++++++ - skipping gene: {} with pubmed_count: {}".format(index, len*list_searches, gene, pubmed_count))
            time.sleep(120)
        except openai.error.ServiceUnavailableError:
            print("\n{}/{} Service unavailable ERROR ++++++++++++++ - skipping gene: {} with pubmed_count: {}".format(index, len*list_searches, gene, pubmed_count))
            time.sleep(120)
        except openai.error.APIConnectionError:
            print("\n{}/{} API Connection ERROR ++++++++++++++ - skipping gene: {} with pubmed_count: {}".format(index, len*list_searches, gene, pubmed_count))
            time.sleep(120)
        except json.decoder.JSONDecodeError:
            print("\n{}/{} Json (bad response) ERROR ++++++++++++++ - skipping gene: {} with pubmed_count: {}".format(index, len*list_searches, gene, pubmed_count))
            time.sleep(120)
        except Exception as e:    
            if e: 
                print(e)   
            print("\n{}/{} Generic ERROR ++++++++++++++ - skipping gene: {} with pubmed_count: {}".format(index, len*list_searches, gene, pubmed_count))
            time.sleep(120)

