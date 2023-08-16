

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
# FILE_TEST = "pubmed23n1159.xml.gz"
FILE_TEST = "pubmed23n1158.xml.gz"
SCHEMA_GPT = "gene_gpt"

# methods

# main
if __name__ == "__main__":
    # get the db connection
    conn = dcc_gpt_lib.get_connection(SCHEMA_GPT)
    skip_processed_files = True
    name_run = "20230815_reference"

    # get the processed abstract files 
    list_files_processed = dcc_gpt_lib.get_db_completed_file_runs(conn=conn, run_name=name_run)

    # get all the files in the pubmed directory
    list_files = dcc_gpt_lib.get_all_files_in_directory(dir_input=DIR_PUBMED)
    list_files = [FILE_TEST]
    print("for files: {}".format(list_files))
    time.sleep(10)

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

            # insert data
            if list_reference and len(list_reference) > 0:
                for id_ref in list_reference:
                    dcc_gpt_lib.insert_db_pubmed_reference(conn=conn, pubmed_id=id_ref, ref_pubmed_id=id_pubmed)
                conn.commit()

        # set file to completed
        dcc_gpt_lib.insert_db_file_run(conn=conn, file_name=file_name, run_name=name_run, completed='Y')



