

# imports
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
from langchain import PromptTemplate,  LLMChain
import os 
import pymysql as mdb
from time import gmtime, strftime
import time



# constants
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
PROMPT_LLM = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in sentences which covers the key points of the text.
```{input_text}```
BULLET POINT SUMMARY:
"""

PROMPT_COMMAND = "Summarize the information related to {} from the following text delimited by triple backquotes.\n"
PROMPT_LLM_GENE = """
```{input_text}```
SUMMARY:
"""


# DB constants
DB_PASSWD = os.environ.get('DB_PASSWD')
NUM_ABSTRACT_LIMIT = 5
SCHEMA_GPT = "gene_gpt"
DB_PAPER_TABLE = "pgpt_paper"
DB_PAPER_ABSTRACT = "pgpt_paper_abtract"

SQL_SELECT_ABSTRACT_BY_TITLE = "select id from {}.pgpt_paper_abstract where title = %s".format(SCHEMA_GPT)
SQL_SELECT_ABSTRACT_LIST_LEVEL_0 = """
select abst.id, abst.abstract 
from {}.pgpt_paper_abstract abst, {}.pgpt_search_paper seapaper 
where abst.document_level = 0 and seapaper.paper_id = abst.pubmed_id and seapaper.search_id = %s limit %s
""".format(SCHEMA_GPT, SCHEMA_GPT, SCHEMA_GPT)
# and abst.id not in (select child_id from {}.pgpt_gpt_paper where search_id = %s) limit %s

SQL_SELECT_ABSTRACT_LIST_LEVEL_HIGHER = """
select distinct abst.id, abst.abstract, abst.document_level
from {}.pgpt_paper_abstract abst, {}.pgpt_gpt_paper gpt
where abst.document_level = %s and gpt.parent_id = abst.id and gpt.search_id = %s
and abst.id not in (select child_id from {}.pgpt_gpt_paper where search_id = %s) limit %s
""".format(SCHEMA_GPT, SCHEMA_GPT, SCHEMA_GPT)

# SQL_INSERT_PAPER = "insert into {}.pgpt_paper (pubmed_id) values(%s)".format(SCHEMA_GPT)
SQL_INSERT_ABSTRACT = "insert into {}.pgpt_paper_abstract (abstract, title, journal_name, document_level) values(%s, %s, %s, %s)".format(SCHEMA_GPT)
SQL_INSERT_GPT_LINK = "insert into {}.pgpt_gpt_paper (search_id, parent_id, child_id, document_level) values(%s, %s, %s, %s)".format(SCHEMA_GPT)
SQL_UPDATE_ABSTRACT_FOR_TOP_LEVEL = "update {}.pgpt_paper_abstract set search_top_level_of = %s where id = %s".format(SCHEMA_GPT)

SQL_SELECT_SEARCHES = "select id, terms, gene from {}.pgpt_search where ready='Y' limit %s".format(SCHEMA_GPT)
SQL_UPDATE_SEARCH_AFTER_SUMMARY = "update {}.pgpt_search set ready = 'N', date_last_summary = sysdate() where id = %s ".format(SCHEMA_GPT)

# methods
def get_model_tokenizer(name, log=False):
    '''
    returns the associated model and tokenizer
    '''
    # initialize
    tokenizer = AutoTokenizer.from_pretrained(name)
    pipeline = transformers.pipeline(
        "text-generation", #task
        model=name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    model = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

    # return
    return model, tokenizer


def summarize(model, gene, command, prompt, text_to_summarize, log=False):
    '''
    summarixes the test input using the model and prompt provided
    '''
    # initialize
    command_gene = command.format(gene)
    prompt_and_command = command_gene + prompt

    if log:
        print("prompt: \n{}".format(prompt_and_command))
    prompt_final = PromptTemplate(template=prompt_and_command, input_variables=["input_text"])
    llm_chain = LLMChain(prompt=prompt_final, llm=model)
    summary = None

    # log
    if log:
        print("using prompt: \n{}".format(prompt_and_command))

    # summarize
    summary = llm_chain.run(text_to_summarize)

    # return
    return summary

def get_list_abstracts(conn, id_search, num_level=0, num_abstracts=NUM_ABSTRACT_LIMIT, log=False):
    '''
    get a list of abstract map objects
    '''
    # initialize
    list_abstracts = []
    cursor = conn.cursor()

    # pick the sql based on level
    if log:
        print("searching for abstracts got input search: {}, doc_level: {}, limit: {}".format(id_search, num_level, num_abstracts))
    if num_level == 0:
        # cursor.execute(SQL_SELECT_ABSTRACT_LIST_LEVEL_0, (id_search, id_search, num_abstracts))
        cursor.execute(SQL_SELECT_ABSTRACT_LIST_LEVEL_0, (id_search, num_abstracts))
    else:
        cursor.execute(SQL_SELECT_ABSTRACT_LIST_LEVEL_HIGHER, (num_level, id_search, id_search, num_abstracts))

    # query 
    db_result = cursor.fetchall()
    for row in db_result:
        paper_id = row[0]
        abstract = row[1]
        list_abstracts.append({"id": paper_id, 'abstract': abstract})

    # return
    return list_abstracts

def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=SCHEMA_GPT)

    # return
    return conn


# main
if __name__ == "__main__":
    # # initialize
    # num_level = 0
    # id_search = 2

    # # get the db connection
    # conn = get_connection()

    # # get the abstracts
    # list_abstracts = get_list_abstracts(conn=conn, id_search=id_search, num_level=num_level, num_abstracts=5, log=True)

    # # get the llm summary
    # str_input = ""
    # if len(list_abstracts) > 1:
    #     # top level is not this level if more than 2 abstracts found at this level
    #     found_top_level = False
    #     for item in list_abstracts:
    #         abstract = item.get('abstract')
    #         print("using abstract: \n{}".format(abstract))
    #         str_input = str_input + " " + abstract

    #     # log
    #     print("using {} for gpt query for level: {} and search: {}".format(len(list_abstracts), num_level, id_search))

    # print("using text: \n{}".format(str_input))

    # # get the model
    # llm_model, tokenizer = get_model_tokenizer(MODEL_NAME)
    # print("got model: {}".format(llm_model))

    # # get the summary
    # summary = summarize(model=llm_model, gene='UBE2NL', command=PROMPT_COMMAND, prompt=PROMPT_LLM_GENE, text_to_summarize=str_input, log=True)
    # print("got summary: \n{}".format(summary))


    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)

    # max length for pipeline indicates max input token 
    pipeline = transformers.pipeline(
        "text-generation", #task
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

    template = """
                Write a concise summary of the following text delimited by triple backquotes.
                Return your response in bullet points which covers the key points of the text.
                ```{text}```
                BULLET POINT SUMMARY:
            """
    template = """
                Write a summary about UBE2NL from the following text delimited by triple backquotes.
                ```{text}```
                BULLET POINT SUMMARY:
            """

    prompt = PromptTemplate(template=template, input_variables=["text"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    text = """ 
Genetic association studies for gastroschisis have highlighted several candidate variants. However, genetic basis in gastroschisis from noninvestigated heritable factors could provide new insights into the human biology for this birth defect. We aim to identify novel gastroschisis susceptibility variants by employing whole exome sequencing (WES) in a Mexican family with recurrence of gastroschisis. We employed WES in two affected half-sisters with gastroschisis, mother, and father of the proband. Additionally, functional bioinformatics analysis was based on SVS-PhoRank and Ensembl-Variant Effect Predictor. The latter assessed the potentially deleterious effects (high, moderate, low, or modifier impact) from exome variants based on SIFT, PolyPhen, dbNSFP, Condel, LoFtool, MaxEntScan, and BLOSUM62 algorithms. The analysis was based on the Human Genome annotation, GRCh37/hg19. Candidate genes were prioritized and manually curated based on significant phenotypic relevance (SVS-PhoRank) and functional properties (Ensembl-Variant Effect Predictor). Functional enrichment analysis was performed using ToppGene Suite, including a manual curation of significant Gene Ontology (GO) biological processes from functional similarity analysis of candidate genes. No single gene-disrupting variant was identified. Instead, 428 heterozygous variations were identified for which SPATA17, PDE4DIP, CFAP65, ALPP, ZNF717, OR4C3, MAP2K3, TLR8, and UBE2NL were predicted as high impact in both cases, mother, and father of the proband. PLOD1, COL6A3, FGFRL1, HHIP, SGCD, RAPGEF1, PKD1, ZFHX3, BCAS3, EVPL, CEACAM5, and KLK14 were segregated among both cases and mother. Multiple interacting background modifiers may regulate gastroschisis susceptibility. These candidate genes highlight a role for development of blood vessel, circulatory system, muscle structure, epithelium, and epidermis, regulation of cell junction assembly, biological/cell adhesion, detection/response to endogenous stimulus, regulation of cytokine biosynthetic process, response to growth factor, postreplication repair/protein K63-linked ubiquitination, protein-containing complex assembly, and regulation of transcription DNA-templated. Considering the likely gene-disrupting prediction results and similar biological pattern of mechanisms, we propose a joint "multifactorial model" in gastroschisis pathogenesis. Cancer is characterized by abnormal growth of cells. Targeting ubiquitin proteins in the discovery of new anticancer therapeutics is an attractive strategy. The present study uses the structure-based drug discovery methods to identify new lead structures, which are selective to the putative ubiquitin-conjugating enzyme E2N-like (UBE2NL). The 3D structure of the UBE2NL was evaluated using homology modeling techniques. The model was validated using standard in silico methods. The hydrophobic pocket of UBE2NL that aids in binding with its natural receptor ubiquitin-conjugating enzyme E2 variant (UBE2V) was identified through protein-protein docking study. The binding site region of the UBE2NL was identified using active site prediction tools. The binding site of UBE2NL which is responsible for cancer cell progression is considered for docking study. Virtual screening study with the small molecular structural database was carried out against the active site of UBE2NL. The ligand molecules that have shown affinity towards UBE2NL were considered for ADME prediction studies. The ligand molecules that obey the Lipinski's rule of five and Jorgensen's rule of three pharmacokinetic properties like human oral absorption etc. are prioritized. The resultant ligand molecules can be considered for the development of potent UBE2NL enzyme inhibitors for cancer therapy. Migraine without aura (MWO) is the most common among migraine group, and is mainly associated with genetic, physical and chemical factors, and hormonal changes. We aimed to identify novel non-synonymous mutations predisposing to the susceptibility to MWO in a Chinese sample using exome sequencing. Four patients with MWO from a family and four non-migraine subjects unrelated with these patients were genotyped using whole-exome sequencing. Bioinformatics analysis was used to screen possible susceptibility gene mutations, which were then verified by PCR. In four patients with MWO, six novel rare non-synonymous mutations were observed, including EDA2R (G170A), UBE2NL (T266G), GBP2 (A907G), EMR1 (C264G), CLCNKB (A1225G), and ARHGAP28 (C413G). It is worth stressing that GBP2 (A907G) was absent in any control subject. Multiple genes predispose to the susceptibility to MWO. ARHGAP28-, EMR1-, and GBP2-encoded proteins may affect angiokinesis, which supports vasogenic theory for the etiological hypothesis of this disease. CLCNKB-encoded protein may affect cell membrane potential, which is consistent with the cortical spreading depression theory. UBE2NL-encoded protein may regulate cellular responses to 5-hydroxytryptamine, which is in accordance with trigeminovascular reflex theory. EDA2R and UBE2NL are located on the X chromosome, which supports that this disease may have gender differences in genetic predisposition. Replication in larger sample size would significantly strengthen these findings. Sporadic Alzheimer disease (SAD) is the most prevalent neurodegenerative disorder. With the development of new generation DNA sequencing technologies, additional genetic risk factors have been described. Here we used various methods to process DNA sequencing data in order to gain further insight into this important disease. We have sequenced the exomes of brain samples from SAD patients and non-demented controls. Using either method, we found a higher number of single nucleotide variants (SNVs), from SAD patients, in genes present at the X chromosome. Using the most stringent method, we validated these variants by Sanger sequencing. Two of these gene variants, were found in loci related to the ubiquitin pathway (UBE2NL and ATXN3L), previously do not described as genetic risk factors for SAD.
    """


    result = llm_chain.run(text)
    print(" ".join(result.split()))
    print(llm_chain.run(text))



