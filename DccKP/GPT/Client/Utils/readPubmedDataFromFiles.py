

# imports
import gzip

# import relative libraries
dir_code = "/home/javaprog/Code/PythonWorkspace/"
import sys
sys.path.insert(0, dir_code + 'MachineLearningPython/DccKP/GPT/')
import dcc_gpt_lib


# constants
DIR_PUBMED = "/scratch/Javaprog/Data/Broad/GPT/Pubmed"
FILE_TEST = "pubmed23n1066.xml.gz"

# methods


# main
if __name__ == "__main__":
    # read in the file
    file_content = ""
    with gzip.open(DIR_PUBMED + "/" + FILE_TEST, 'r') as f:
        file_content = f.read()

    # get the json from the xml
    list_pubmed = dcc_gpt_lib.get_pubmed_article_list(xml_input=file_content, log=False)
    print("got paper list of size: {}".format(len(list_pubmed)))

    # get the list of data
    for item in list_pubmed:
        text_abstract, title, journal, year, id_pubmed = dcc_gpt_lib.get_paper_data_from_map(item)
        print("got paper: {} - {} - {} \n {}".format(id_pubmed, year, journal, text_abstract))
    print("got paper list of size: {}".format(len(list_pubmed)))
