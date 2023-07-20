

# imports
import openai
import os 
import pymysql as mdb
from time import gmtime, strftime
import time

# constants
STR_INPUT1 = "We performed collapsing analyses on 454,796 UK Biobank (UKB) exomes to detect gene-level associations with diabetes. Recessive carriers of nonsynonymous variants in  were 30% less likely to develop diabetes ( = 5.7 × 10) and had lower glycosylated hemoglobin (β = -0.14 SD units,  = 1.1 × 10). These associations were independent of body mass index, suggesting protection against insulin resistance even in the setting of obesity. We replicated these findings in 96,811 Admixed Americans in the Mexico City Prospective Study ( < 0.05)Moreover, the protective effect of  variants was stronger in individuals who did not carry the Latino-enriched  risk haplotype ( = 6.0 × 10). Separately, we identified a Finnish-enriched  protein-truncating variant associated with decreased odds of both type 1 and type 2 diabetes ( < 0.05) in FinnGen. No adverse phenotypes were associated with protein-truncating  variants in the UKB, supporting this gene as a therapeutic target for diabetes."
STR_INPUT2 = "A major goal in human genetics is to use natural variation to understand the phenotypic consequences of altering each protein-coding gene in the genome. Here we used exome sequencing to explore protein-altering variants and their consequences in 454,787 participants in the UK Biobank study. We identified 12 million coding variants, including around 1 million loss-of-function and around 1.8 million deleterious missense variants. When these were tested for association with 3,994 health-related traits, we found 564 genes with trait associations at P ≤ 2.18 × 10. Rare variant associations were enriched in loci from genome-wide association studies (GWAS), but most (91%) were independent of common variant signals. We discovered several risk-increasing associations with traits related to liver disease, eye disease and cancer, among others, as well as risk-lowering associations for hypertension (SLC9A3R2), diabetes (MAP3K15, FAM234A) and asthma (SLC27A3). Six genes were associated with brain imaging phenotypes, including two involved in neural development (GBE1, PLD1). Of the signals available and powered for replication in an independent cohort, 81% were confirmed; furthermore, association signals were generally consistent across individuals of European, Asian and African ancestry. We illustrate the ability of exome sequencing to identify gene-trait associations, elucidate gene function and pinpoint effector genes that underlie GWAS signals at scale."
STR_INPUT3 = "Mitogen-activated protein kinases (MAP kinases) are functionally connected kinases that regulate key cellular process involved in kidney disease such as all survival, death, differentiation and proliferation. The typical MAP kinase module is composed by a cascade of three kinases: a MAP kinase kinase kinase (MAP3K) that phosphorylates and activates a MAP kinase kinase (MAP2K) which phosphorylates a MAP kinase (MAPK). While the role of MAPKs such as ERK, p38 and JNK has been well characterized in experimental kidney injury, much less is known about the apical kinases in the cascade, the MAP3Ks. There are 24 characterized MAP3K (MAP3K1 to MAP3K21 plus RAF1, BRAF and ARAF). We now review current knowledge on the involvement of MAP3K in non-malignant kidney disease and the therapeutic tools available. There is in vivo interventional evidence clearly supporting a role for MAP3K5 (ASK1) and MAP3K14 (NIK) in the pathogenesis of experimental kidney disease. Indeed, the ASK1 inhibitor Selonsertib has undergone clinical trials for diabetic kidney disease. Additionally, although MAP3K7 (MEKK7, TAK1) is required for kidney development, acutely targeting MAP3K7 protected from acute and chronic kidney injury; and targeting MAP3K8 (TPL2/Cot) protected from acute kidney injury. By contrast MAP3K15 (ASK3) may protect from hypertension and BRAF inhibitors in clinical use may induced acute kidney injury and nephrotic syndrome. Given their role as upstream regulators of intracellular signaling, MAP3K are potential therapeutic targets in kidney injury, as demonstrated for some of them. However, the role of most MAP3K in kidney disease remains unexplored."
KEY_CHATGPT = os.environ.get('CHAT_KEY')
openai.api_key = KEY_CHATGPT
MODEL_CHATGPT="gpt-3.5-turbo-0301"
MODEL_PROMPT_SUMMARIZE = "summarize the following in 200 words: \n{}"

SEARCH_ID=1
GPT_PROMPT = "summarize the information related to {} from the information below:\n"

# db constants
DB_PASSWD = os.environ.get('DB_PASSWD')
NUM_ABSTRACT_LIMIT = 5
SCHEMA_GPT = "gene_gpt"
DB_PAPER_TABLE = "pgpt_paper"
DB_PAPER_ABSTRACT = "pgpt_paper_abtract"
SQL_SELECT_ABSTRACT_BY_TITLE = "select id from {}.pgpt_paper_abstract where title = %s".format(SCHEMA_GPT)
SQL_SELECT_ABSTRACT_LIST = "select id, abstract from {}.pgpt_paper_abstract where document_level = %s and id not in (select child_id from {}.pgpt_gpt_paper where search_id = %s) limit %s".format(SCHEMA_GPT, SCHEMA_GPT)
# SQL_INSERT_PAPER = "insert into {}.pgpt_paper (pubmed_id) values(%s)".format(SCHEMA_GPT)
SQL_INSERT_ABSTRACT = "insert into {}.pgpt_paper_abstract (abstract, title, journal_name, document_level) values(%s, %s, %s, %s)".format(SCHEMA_GPT)
SQL_INSERT_GPT_LINK = "insert into {}.pgpt_gpt_paper (search_id, parent_id, child_id, document_level) values(%s, %s, %s, %s)".format(SCHEMA_GPT)

# methods
def call_chatgpt(str_query, log=False):
    '''
    makes the api call to chat gpt service
    '''
    # initialize
    str_result = ""
    list_conversation = []

    # build the payload
    list_conversation.append({'role': 'system', 'content': MODEL_PROMPT_SUMMARIZE.format(str_query)})
    if log:
        print("using chat input: {}".format(list_conversation))

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

def get_list_abstracts(conn, id_search, num_level=0, num_abstracts=NUM_ABSTRACT_LIMIT, log=False):
    '''
    get a list of abstract map objects
    '''
    # initialize
    list_abstracts = []
    cursor = conn.cursor()

    # query 
    cursor.execute(SQL_SELECT_ABSTRACT_LIST, (num_level, id_search, num_abstracts))
    db_result = cursor.fetchall()
    for row in db_result:
        paper_id = row[0]
        abstract = row[1]
        list_abstracts.append({"id": paper_id, 'abstract': abstract})

    # return
    return list_abstracts

def insert_gpt_results(conn, id_search, num_level, list_abstracts, gpt_abstract, log=False):
    '''
    insert the gpt list
    '''
    # intialize
    id_result = 0
    cursor = conn.cursor()
    level_document = num_level + 1

    # insert the result
    str_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    title = "GPT - {}".format(str_time)
    journal_name = "GPT-3.5"
    if log:
        print("generating GPT entry: {}".format(title))
    cursor.execute(SQL_INSERT_ABSTRACT, (gpt_abstract, title, journal_name, level_document))
    conn.commit()

    # get the id
    cursor.execute(SQL_SELECT_ABSTRACT_BY_TITLE, (title))
    db_result = cursor.fetchall()
    if log:
        print("found parent_id result: {}".format(db_result))
    if db_result:
        id_result = db_result[0][0]

    # insert the links
    for item in list_abstracts:
        cursor.execute(SQL_INSERT_GPT_LINK, (id_search, id_result, item.get('id'), level_document))

    # commit
    conn.commit()

    # return
    return id_result

def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=SCHEMA_GPT)

    # return
    return conn


# main
if __name__ == "__main__":
    # initiliaze
    list_input = [STR_INPUT1, STR_INPUT2, STR_INPUT3]
    str_input = " ".join(list_input)    
    num_level = 2
    id_search = 0
    num_abstracts = 7
    gpt_prompt = GPT_PROMPT.format("PPARG")
    print("using GPT prompt: {}".format(gpt_prompt))

    # # get the chat gpt response
    # str_chat = call_chatgpt(str_input, log=True)
    # print("got chat gpt string: {}".format(str_chat))


    # get the connection
    conn = get_connection()

    # get 5 papers
    for i in range(100000):
        str_input = gpt_prompt
        list_abstracts = get_list_abstracts(conn=conn, id_search=id_search, num_level=num_level, num_abstracts=num_abstracts)
        if len(list_abstracts) > 2:
            for item in list_abstracts:
                abstract = item.get('abstract')
                print("using abstract: \n{}".format(abstract))
                str_input = str_input + " " + abstract

            # get the chat gpt response
            str_chat = call_chatgpt(str_input, log=False)
            print("\n\ngot chat gpt string: {}".format(str_chat))

            # insert results and links
            insert_gpt_results(conn=conn, id_search=id_search, num_level=num_level, list_abstracts=list_abstracts, gpt_abstract=str_chat, log=True)
            time.sleep(30)
        else:
            print("\n\n\n\nNo more articles")
            break