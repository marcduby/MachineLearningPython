

# imports
import gzip
import time 
import os

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
import dcc_pubmed_lib


# constants
DIR_PUBMED = "/scratch/Javaprog/Data/Broad/Pubmed"
if ENV_DIR_PUBMED:
    DIR_PUBMED = ENV_DIR_PUBMED
FILE_TEST = "pubmed23n1166.xml.gz"
SCHEMA_GPT = "pubmed_gen"

# methods
def get_all_files_in_directory(dir_input, log=False):
    '''
    get all the files in a directory
    '''
    list_files = []

    # get all the files
    # Iterate over directory contents
    for entry in os.scandir(dir_input):
        # If entry is a file, store it
        if entry.is_file():
            list_files.append(entry.name) 

    # return
    return list_files

# main
if __name__ == "__main__":
    # settings
    name_process = "load_abstract_20240104"

    # get the db connection
    conn = dcc_gpt_lib.get_connection(SCHEMA_GPT)
    skip_processed_files = True

    # get the processed abstract files 
    list_files_processed = dcc_pubmed_lib.get_files_processed_list(conn=conn, process_name=name_process)
    print("for process: {} got file processed of fize: {}".format(name_process, len(list_files_processed)))

    # get all the files in the pubmed directory
    list_files = get_all_files_in_directory(dir_input=DIR_PUBMED)
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
        time.sleep(5)

        # get the list of data
        for item in list_pubmed:
            text_abstract, title, journal, year, id_pubmed = dcc_gpt_lib.get_paper_data_from_map(item)
            # print("got paper: {} - {} - {} \n {}".format(id_pubmed, year, journal, text_abstract))

            # find if the papert has not been loaded (could use cache, but want to check for duplicates in files)
            # only called if needed
            id_row = dcc_pubmed_lib.get_db_if_pubmed_downloaded_general(conn=conn, pubmed_id=id_pubmed, log=False)
            if not id_row:
                # if not loaded and needed, then insert
                dcc_pubmed_lib.insert_db_paper_abstract_general(conn=conn, pubmed_id=id_pubmed, abstract=text_abstract, journal=journal, title=title, 
                                                    year=year, file_name=file_name, log=True)

        # add the file to the processed table
        dcc_pubmed_lib.insert_db_file_processed_general(conn=conn, file_name=file_name, process_name=name_process)

        # log total papers for this file
        print("{}/{} - for file: {}, got paper list of size: {}\n\n".format(index, len(list_files), file_name, len(list_pubmed)))


