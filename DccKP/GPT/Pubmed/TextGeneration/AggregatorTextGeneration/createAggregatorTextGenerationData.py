
# imports
import os 
import pymysql as mdb
import glob 
import io
import json

# constants
DB_PASSWD = os.environ.get('DB_PASSWD')
SCHEMA_GPT = "tran_upkeep"
DIR_DATA = "/home/javaprog/Data/Broad/GPT/Data/TextGenerationAggregator"
FILE_DATA = "{}/text_generation_data_aggregator.json".format(DIR_DATA)
FILE_KEYWORDS = "{}/text_generation_aggregator_keywords.json".format(DIR_DATA)

FLOAT_PROB_CUTOFF=0.2

# sql statements
SQL_SELECT_GENEBASS = """
select assoc.id, assoc.gene, assoc.phenotype_genepro_name as phenotype_name, assoc.beta, assoc.probability as prob 
from {}.data_genebass_gene_phenotype assoc
where assoc.probability > {} and assoc.phenotype_genepro_name is not null
""".format(SCHEMA_GPT, FLOAT_PROB_CUTOFF)

SQL_SELECT_600k = """
select assoc.id, assoc.gene_code as gene, pheno.phenotype_translator_name as phenotype_name, assoc.beta, assoc.probability_calculated as prob 
from {}.data_600k_gene_phenotype assoc, {}.data_600k_phenotype_ontology pheno
where assoc.probability_calculated > {} and assoc.phenotype_code = pheno.phenotype_code and assoc.mask = 'LoF_HC'
""".format(SCHEMA_GPT, SCHEMA_GPT, FLOAT_PROB_CUTOFF)

SQL_SELECT_MAGMA = """
select assoc.id, assoc.gene_code as gene, pheno.phenotype_name as phenotype_name, assoc.abf_probability_combined as prob 
from {}.agg_gene_phenotype assoc, {}.agg_aggregator_phenotype pheno
where assoc.abf_probability_combined > {} and assoc.phenotype_code = pheno.phenotype_id
""".format(SCHEMA_GPT, SCHEMA_GPT, FLOAT_PROB_CUTOFF)

SQL_SELECT_PATHWAY_GENES = """
select assoc.id, assoc.gene_code as gene, pathway.pathway_name as pathway_name, pathway.ontology_id
from {}.data_pathway_genes assoc, {}.data_pathway pathway
where assoc.pathway_id = pathway.id and pathway.ontology_id is not null
""".format(SCHEMA_GPT, SCHEMA_GPT, FLOAT_PROB_CUTOFF)

SQL_SELECT_PATHWAY_ASSOCIATIONs = """
select assoc.id, assoc.gene_code as gene, pathway.pathway_name as pathway_name, pathway.ontology_id
from {}.data_pathway_genes assoc, {}.data_pathway pathway
where assoc.pathway_id = pathway.id and pathway.ontology_id is not null
""".format(SCHEMA_GPT, SCHEMA_GPT, FLOAT_PROB_CUTOFF)

SQL_SELECT_PHENOTYPE_PHENOTYPE_ASSOCIATIONS = """
select assoc.id, pheno1.phenotype_name as phenotype_name1, pheno2.phenotype_name as phenotype_name2
from {}.agg_phenotype_phenotype assoc, {}.agg_aggregator_phenotype pheno1, {}.agg_aggregator_phenotype pheno2
where assoc.phenotype_subj_code = pheno1.phenotype_id and assoc.phenotype_obj_code = pheno2.phenotype_id
""".format(SCHEMA_GPT, SCHEMA_GPT, SCHEMA_GPT)

# SQL_SELECT_600k = "select id, gene, phenotype as phenotype_name, beta, probability_calculated as prob from {}.data_600k_gene_phenotype where probability_calculated > {}".format(SCHEMA_GPT, FLOAT_PROB_CUTOFF)

# methods
def create_sentence_from_beta_prob_row(row, log=False):
    '''
    creates a sentence string from the db row of a beta prob table
    '''
    sentence1 = ""
    sentence_template1 = "gene {} has a {} genetic association with {}"
    sentence2 = ""
    sentence_template2 = "{} has a {} genetic association with gene {}"
    direction = ""
    direction_template = " in the {} direction"

    # get the data
    if row.get('gene') and row.get('phenotype_name') and row.get('prob'):
        adjective = get_association_adjective(row.get('prob'))
        sentence1 = sentence_template1.format(row.get('gene'), adjective, row.get('phenotype_name'))
        sentence2 = sentence_template2.format(row.get('phenotype_name'), adjective, row.get('gene'))

        if row.get('beta'):
            # print("'{}'".format(row.get('beta')))
            if row.get('beta') < 0:
                direction = direction_template.format("negative")
            else:
                direction = direction_template.format("positive")
            sentence1 = sentence1 + direction
            sentence2 = sentence2 + direction

    # return
    return [sentence1, sentence2]

def create_sentence_from_gene_pathway_row(row, log=False):
    '''
    creates a sentence string from the db row of a gene pathways
    '''
    sentence = ""
    sentence_template = "gene {} is part of pathway {} with curie {}"
    sentence2 = ""
    sentence2_template = "pathway {} with curie {} contains gene {}"

    # get the data
    if row.get('gene') and row.get('pathway_name') and row.get('ontology_id'):
        sentence = sentence_template.format(row.get('gene'), row.get('pathway_name'), row.get('ontology_id'))
        sentence2 = sentence2_template.format(row.get('pathway_name'), row.get('ontology_id'), row.get('gene'))

    # return
    return [sentence, sentence2]

def get_association_adjective(float_prob, log=False):
    '''
    creates the qualified statement for the probability given
    '''            
    adjective = "weak"

    if float_prob > 0.75:
        adjective = "strong"
    elif float_prob > 0.5:
        adjective = "firm"
    elif float_prob > 0.25:
        adjective = "mild"

    # return
    return adjective


def create_json_dataset_file(list_input, file_name, log=False):
    # save to json
    with open(file_name, "w+") as f:
        json.dump(list_input, f)

    print("wrote out: {} size list to: {}".format(len(list_input), file_name))

def create_gpt_sentence_list(list_input, str_start="<start> ", str_end=" <end>", log=False):
    '''
    creates a list of user/bot conversations
    '''
    list_result = []

    # loop through array
    for item in list_input:
        str_temp = str_start + item + str_end
        list_result.append(str_temp)

    # return
    return list_result

def get_list_of_gene_association_sentences(conn, string_select, has_beta = True, log=False):
    '''
    retrieves the prob beta data from the DB
    '''
    cursor = conn.cursor()
    list_sentences = []
    set_keywords = set()

    # run the qery
    cursor.execute(string_select)

    # get the results
    db_results = cursor.fetchall()
    print("got DB results of size: {}".format(len(db_results)))
    for row in db_results:
        if has_beta:
            map_row = {'gene': row[1], 'phenotype_name': row[2], 'beta': row[3], 'prob': row[4]}
        else: 
            map_row = {'gene': row[1], 'phenotype_name': row[2], 'prob': row[3]}

        # get the senetnces
        list_temp = create_sentence_from_beta_prob_row(map_row)
        for sentence in list_temp:
            if len(sentence) > 5:
                list_sentences.append(sentence)

        if map_row.get('gene'):
            set_keywords.add(map_row.get('gene'))
        if map_row.get('phenotype_name'):
            set_keywords.update(map_row.get('phenotype_name').split(" "))

        if log:
            # print("db: {}".format(row))
            print("sentence: {}".format(list_temp))

    # return
    return list_sentences, set_keywords

def get_list_of_gene_pathway_sentences(conn, string_select, log=False):
    '''
    retrieves the gene pathway data from the DB
    '''
    cursor = conn.cursor()
    list_sentences = []
    set_keywords = set()
    
    # run the qery
    cursor.execute(string_select)

    # get the results
    db_results = cursor.fetchall()
    print("got DB results of size: {}".format(len(db_results)))
    for row in db_results:
        map_row = {'gene': row[1], 'pathway_name': row[2], 'ontology_id': row[3]}

        list_temp = create_sentence_from_gene_pathway_row(map_row)
        for sentence in list_temp:
            if len(sentence) > 5:
                list_sentences.append(sentence)

            if log:
                print("db: {}".format(row))
                print("sentence: {}".format(sentence))

        if map_row.get('gene'):
            set_keywords.add(map_row.get('gene'))
        if map_row.get('pathway_name'):
            set_keywords.update(map_row.get('pathway_name').split(" "))
        if map_row.get('ontology_id'):
            set_keywords.add(map_row.get('ontology_id'))

    # return
    return list_sentences, set_keywords

def get_list_of_phenotype_phenotype_sentences(conn, string_select, log=False):
    '''
    retrieves the phenotype phenotype data from the DB
    '''
    cursor = conn.cursor()
    list_sentences = []
    set_keywords = set()
    sentence_template = "{} has a genetic association with {}"

    # run the qery
    cursor.execute(string_select)

    # get the results
    db_results = cursor.fetchall()
    print("got DB results of size: {}".format(len(db_results)))
    for row in db_results:
        # map_row = {'phenotype_name1': row[1], 'phenotype_name2': row[2]}

        # create the sentence
        sentence = sentence_template.format(row[1], row[2])
        if len(sentence) > 5:
            list_sentences.append(sentence)
        sentence = sentence_template.format(row[2], row[1])
        if len(sentence) > 5:
            list_sentences.append(sentence)

            if log:
                print("db: {}".format(row))
                print("sentence: {}".format(sentence))

        set_keywords.update(row[1].split(" "))
        set_keywords.update(row[2].split(" "))

    # return
    return list_sentences, set_keywords


def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=SCHEMA_GPT)

    # return
    return conn


# main
if __name__ == "__main__":
    # initalize
    list_sentences = []
    num_count = 0

    # get the list of abstracts
    conn = get_connection()

    # create the keyword set
    set_keywords = {'mild', 'strong', 'firm', 'weak', 'genetic', 'association', 'gene', 'pathway', 'direction', 'positive', 'negative', 'curie'}

    # get genebass data 
    list_temp, set_temp = get_list_of_gene_association_sentences(conn, SQL_SELECT_GENEBASS, log=False)
    print("to process, got genebass list of size: {} with keyword set of size: {}".format(len(list_temp), len(set_temp)))
    list_sentences = list_sentences + list_temp
    set_keywords = set_keywords.union(set_temp)
    print("after genebass, sentence list of size: {} with keyword set of size: {}".format(len(list_sentences), len(set_keywords)))

    # get 600k data
    list_temp, set_temp = get_list_of_gene_association_sentences(conn, SQL_SELECT_600k, log=False)
    print("to process, got 600k gene phenotype list of size: {} with keywords size: {}".format(len(list_temp), len(set_temp)))
    list_sentences = list_sentences + list_temp
    set_keywords = set_keywords.union(set_temp)
    print("after gene pathways, sentence list of size: {} with keyword set of size: {}".format(len(list_sentences), len(set_keywords)))

    # get magma data
    list_temp, set_temp = get_list_of_gene_association_sentences(conn, SQL_SELECT_MAGMA, has_beta=False, log=False)
    print("to process, got magma gene phenotype list of size: {} with keywords size: {}".format(len(list_temp), len(set_temp)))
    list_sentences = list_sentences + list_temp
    set_keywords = set_keywords.union(set_temp)
    print("after gene pathways, sentence list of size: {} with keyword set of size: {}".format(len(list_sentences), len(set_keywords)))

    # get gene pathway data
    list_temp, set_temp = get_list_of_gene_pathway_sentences(conn, SQL_SELECT_PATHWAY_GENES, log=False)
    print("to process, got pathway gene list of size: {} with keywords size: {}".format(len(list_temp), len(set_temp)))
    list_sentences = list_sentences + list_temp
    set_keywords = set_keywords.union(set_temp)
    print("after gene pathways, sentence list of size: {} with keyword set of size: {}".format(len(list_sentences), len(set_keywords)))

    # get pathway disease data
    list_temp, set_temp = get_list_of_phenotype_phenotype_sentences(conn, SQL_SELECT_PHENOTYPE_PHENOTYPE_ASSOCIATIONS, log=False)
    print("to process, got phanotype/phenotype list of size: {} with keywords size: {}".format(len(list_temp), len(set_temp)))
    list_sentences = list_sentences + list_temp
    set_keywords = set_keywords.union(set_temp)
    print("after phenotype/phenotype, sentence list of size: {} with keyword set of size: {}".format(len(list_sentences), len(set_keywords)))

    # for each abstract
    # for sentence in list_sentences:
    #     num_count = num_count + 1
    #     print(sentence)

    # annotate the sentences
    list_result = create_gpt_sentence_list(list_sentences)

    # write out the conversations
    create_json_dataset_file(list_result, FILE_DATA)
    create_json_dataset_file(list(set_keywords), FILE_KEYWORDS)


