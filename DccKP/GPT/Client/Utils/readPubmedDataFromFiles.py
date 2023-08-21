

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


# constants
DIR_PUBMED = "/scratch/Javaprog/Data/Broad/GPT/Pubmed"
if ENV_DIR_PUBMED:
    DIR_PUBMED = ENV_DIR_PUBMED
FILE_TEST = "pubmed23n1166.xml.gz"
SCHEMA_GPT = "gene_gpt"

# methods
def get_all_fiels_in_directory(dir_input, log=False):
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
    # get the db connection
    conn = dcc_gpt_lib.get_connection(SCHEMA_GPT)
    skip_processed_files = False

    # get the processed abstract files 
    list_files_processed = dcc_gpt_lib.get_db_abstract_files_processed(conn=conn)

    # load the cache id set 
    set_pubmed_id = dcc_gpt_lib.get_db_all_pubmed_ids(conn=conn)
    print("for to download pubmed id set of size: {}".format(len(set_pubmed_id)))

    # get all the files in the pubmed directory
    list_files = get_all_fiels_in_directory(dir_input=DIR_PUBMED)
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
        for item in list_pubmed:
            text_abstract, title, journal, year, id_pubmed = dcc_gpt_lib.get_paper_data_from_map(item)
            # print("got paper: {} - {} - {} \n {}".format(id_pubmed, year, journal, text_abstract))

            # find if the paper is needed (from cache), called for every entry
            if id_pubmed in set_pubmed_id:
                # sleep for now
                # time.sleep(0.01)

                # find if the papert has not been loaded (could use cache, but want to check for duplicates in files)
                # only called if needed
                id_row = dcc_gpt_lib.get_db_if_pubmed_downloaded(conn=conn, pubmed_id=id_pubmed, log=False)
                if not id_row:
                    # if not loaded and needed, then insert
                    dcc_gpt_lib.insert_db_paper_abstract(conn=conn, pubmed_id=id_pubmed, abstract=text_abstract, journal=journal, title=title, 
                                                        year=year, document_level=0, file_name=file_name, log=True)

        # log total papers
        print("for file: {}, got paper list of size: {}\n\n".format(file_name, len(list_pubmed)))


