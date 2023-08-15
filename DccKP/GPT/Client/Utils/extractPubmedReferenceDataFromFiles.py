

# imports
import gzip
import time 
import os
import json 

# import relative libraries
dir_code = "/home/javaprog/Code/PythonWorkspace/"
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/GPT/')
import dcc_gpt_lib


# constants
DIR_PUBMED = "/scratch/Javaprog/Data/Broad/GPT/Pubmed"
FILE_TEST = "pubmed23n1159.xml.gz"
SCHEMA_GPT = "gene_gpt"

# methods

# main
if __name__ == "__main__":
    # get the db connection
    conn = dcc_gpt_lib.get_connection(SCHEMA_GPT)
    skip_processed_files = False

    # get the processed abstract files 
    list_files_processed = dcc_gpt_lib.get_db_abstract_files_processed(conn=conn)

    # load the cache id set 
    set_pubmed_id = dcc_gpt_lib.get_db_all_pubmed_ids(conn=conn)
    print("for to download pubmed id set of size: {}".format(len(set_pubmed_id)))

    # get all the files in the pubmed directory
    # list_files = dcc_gpt_lib.get_all_files_in_directory(dir_input=DIR_PUBMED)
    list_files = [FILE_TEST]
    # print("for files: {}".format(list_files))
    # time.sleep(100)

    # loop through files
    # file_name = FILE_TEST
    for index, file_name in enumerate(list_files):
        # see if file not processed yet
        if skip_processed_files and file_name in list_files_processed:
            print("{}/{} - skipping processed file: {}".format(index, len(list_files), file_name))
            continue
        
        # read in the file
        file_content = ""
        with gzip.open(DIR_PUBMED + "/" + file_name, 'r') as f:
            file_content = f.read()

        # get the json from the xml
        print("{}/{} - reading file: {}".format(index, len(list_files), file_name))
        list_pubmed = dcc_gpt_lib.get_pubmed_article_list(xml_input=file_content, log=False)
        print("for file: {}, got paper list of size: {}".format(file_name, len(list_pubmed)))
        time.sleep(3)

        # get the list of data
        for jindex, item in enumerate(list_pubmed):
            id_pubmed, list_reference = dcc_gpt_lib.get_paper_references_from_map(item, log=False)

            # test
            # if id_pubmed == 36253005:
            if jindex == 0:
                print("got {} paper json: \n{}".format(id_pubmed, json.dumps(item, indent=1)))
            print("for: {} got list of references: {}".format(id_pubmed, list_reference))




