
# imports
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
import dcc_langchain_lib

# constants

# methods

# main
if __name__ == "__main__":
    # initialize
    num_abstracts_per_summary = 5
    max_per_level = 5
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
        # time.sleep(5)

        # get all the abstracts for the document level and run
        list_abstracts = dcc_gpt_lib.get_list_abstracts(conn=conn, id_search=id_search, id_run=id_run, num_level=num_level, num_abstracts=max_per_level, log=True)
        list_text = [t.get('abstract') for t in list_abstracts]

        # print each abstract
        for abs in list_text:
            print("abstract: \n{}\n".format(abs))

        # create the docs fromt he abstract list
        list_docs = dcc_langchain_lib.create_docs_list_from_text_list(list_text=list_text)

        # get the model
        llm = dcc_langchain_lib.load_local_llama_model(file_model=dcc_langchain_lib.FILE_LLAMA2_13B_CPU)

        # get the summary chain
        chain_summary = dcc_langchain_lib.get_summarize_chain(llm=llm, verbose=True)

        # run the chain
        response = chain_summary.run(list_docs)
        print("==================================================")
        print("got response: \n{}".format(response))

        break
        time.sleep(3)




