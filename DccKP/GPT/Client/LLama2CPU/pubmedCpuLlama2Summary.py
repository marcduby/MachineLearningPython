

# imports
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import time 
import os

# for AWS
ENV_DIR_CODE = os.environ.get('DIR_CODE')
ENV_DIR_PUBMED = os.environ.get('DIR_PUBMED')

# local imports
dir_code = "/home/javaprog/Code/PythonWorkspace/"
if ENV_DIR_CODE:
    dir_code = ENV_DIR_CODE
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/GPT/')
import dcc_gpt_lib

# constants
DIR_DATA = "/home/javaprog/Data/"
DIR_DOCS = DIR_DATA + "ML/Llama2Test/Genetics/Docs"
DIR_VECTOR_STORE = DIR_DATA + "ML/Llama2Test/Genetics/VectorStore"
FILE_MODEL = DIR_DATA + "ML/Llama2Test/Model/llama-2-7b-chat.ggmlv3.q8_0.bin"
PROMPT = """Use the following piece of information to anser the user's question.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing elase.
Helpful answer:
"""
GPT_PROMPT = """
Below are the abstracts from different research papers on gene {gene}. 
Please read through the abstracts and as a genetics researcher write a 100 word summary that synthesizes the key findings of the papers on the biology of gene {gene}
{abstracts}
"""

# methods
def set_prompt(prompt, log=False):
    '''
    returns the prompt to use
    '''
    result_prompt = PromptTemplate(template=prompt, input_variables=['gene', 'abstracts'])

    return result_prompt

def load_llm(file_model, log=False):
    if log:
        print("loading model: {}".format(file_model))

    llm = CTransformers(
        model=file_model,
        model_type = "llama",
        max_new_tokens = 512,
        # temperature = 0.1
        temperature = 0.5
    )

    if log:
        print("loaded model from: {}".format(file_model))

    return llm

def get_qa_chain(llm, prompt, db=None, log=False):
    '''
    get the langchain
    '''
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        # retriever = db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_chain

def get_qa_bot(dir_db, file_model, prompt, log=False):
    # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    # db = FAISS.load_local(dir_db, embeddings)

    llm = load_llm(file_model=file_model, log=log)

    prompt_qa = set_prompt(prompt=prompt, log=log)

    chain_qa = get_qa_chain(llm=llm, prompt=prompt_qa, log=log)

    return chain_qa

def get_inference(question, chain_qa, log=False):
    '''
    do the llm inference
    '''
    if log:
        print("doing llm inference using query: {}".format(question))
    result = chain_qa({'query': question})

    # if log:
    #     print("got result: {}".format(result))

    # return
    return result

def get_inference_gene_abstracts(gene, abstracts, chain_qa, log=False):
    '''
    do the llm inference
    '''
    if log:
        print("doing llm inference using gene: {} and abstcats: \n{}".format(gene, abstracts))
    result = chain_qa({'gene': gene, 'abstracts': abstracts})

    # if log:
    #     print("got result: {}".format(result))

    # return
    return result

# main
if __name__ == "__main__":
    # initialize
    # dir_db = DIR_VECTOR_STORE
    file_model = FILE_MODEL
    prompt = PROMPT
    num_abstracts_per_summary = 5
    max_per_level = 50
    id_run = 7
    num_level = 0

    # get the connection
    conn = dcc_gpt_lib.get_connection()

    # get the name and prompt of the run
    _, name_run, prompt_run = dcc_gpt_lib.get_db_run_data(conn=conn, id_run=id_run)
    print("got run: {} with prompt: \n'{}'\n".format(name_run, prompt_run))

    # get the list of searches
    list_searches = dcc_gpt_lib.get_db_list_ready_searches(conn=conn, num_searches=1)
    # get the best abstracts for gene

    for search in list_searches:
        id_search = search.get('id')
        id_top_level_abstract = -1
        gene = search.get('gene')

        # log
        print("\nprocessing search: {} for gene: {} for run id: {} of name: {}".format(id_search, gene, id_run, name_run))
        time.sleep(5)

        # get all the abstracts for the document level and run
        list_abstracts = dcc_gpt_lib.get_list_abstracts(conn=conn, id_search=id_search, id_run=id_run, num_level=num_level, num_abstracts=max_per_level, log=True)

        for i in range(0, len(list_abstracts), num_abstracts_per_summary):
            list_sub = list_abstracts[i : i + num_abstracts_per_summary] 

            # for the sub list
            str_abstracts = ""
            for item in list_sub:
                abstract = item.get('abstract')
                word_count = len(abstract.split())
                print("using abstract with count: {} and content: \n{}".format(word_count, abstract))
                str_abstracts = str_abstracts + "\n" + abstract

            # log
            print("using abstract count: {} for gpt query for level: {} and search: {}".format(len(list_sub), num_level, id_search))

            # build the prompt
            str_prompt = prompt_run.format(gene, gene, str_abstracts)

            # get the chat gpt response
            print("creating langchain")
            chain_qa = get_qa_bot(dir_db=None, file_model=file_model, prompt=GPT_PROMPT, log=True)

            # do inference
            response = get_inference_gene_abstracts(gene=gene, abstracts=str_abstracts, chain_qa=chain_qa, log=True)
            print("got response: \n\n{}".format(response))
            # print("using GPT prompt: \n{}".format(str_prompt))

            # insert results and links
            # dcc_gpt_lib.insert_gpt_results(conn=conn, id_search=id_search, num_level=num_level, list_abstracts=list_abstracts, 
            #                                 gpt_abstract=str_chat, id_run=id_run, name_run=name_run, log=True)
            # time.sleep(30)
            break
            time.sleep(3)


    # # build the abstracts
    # # get the langchain
    # print("creating langchain")
    # chain_qa = get_qa_bot(dir_db=None, file_model=file_model, prompt=prompt, log=True)

    # # do inference
    # question = "what is the translator solution"
    # response = get_inference(question=question, chain_qa=chain_qa, log=True)
    # print("got response: \n\n{}".format(response))

    # # do inference
    # question = "what is an ARA"
    # response = get_inference(question=question, chain_qa=chain_qa, log=True)
    # print("got response: \n\n{}".format(response))

    # # do inference
    # question = "what are an KPs"
    # response = get_inference(question=question, chain_qa=chain_qa, log=True)
    # print("got response: \n\n{}".format(response))


