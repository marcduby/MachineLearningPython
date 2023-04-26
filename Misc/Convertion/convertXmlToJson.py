
import os 
import xmltodict
import re
import glob 
import io
import json 

# constants 
FILE_TEST_XML = "/home/javaprog/Code/PythonWorkspace/MachineLearningPython/DccKP/GPT/Pubmed/test.xml"
FILE_TEST_JSON = "/home/javaprog/Code/PythonWorkspace/MachineLearningPython/DccKP/GPT/Pubmed/test.json"


def parse_xml_file(xmlfile, log=False):
    '''
    parses the xml file into map
    '''
    xml_doc = None 

    # parse the file
    with open(xmlfile) as xml_input:
        xml_doc = xmltodict.parse(xml_input.read())

    # return
    return xml_doc

if __name__ == "__main__":
    # get the dict of the xml file 
    json_doc = parse_xml_file(FILE_TEST_XML)

    # print
    print(json_doc)

    # save to json
    with open(FILE_TEST_JSON, "w+") as f:
        json.dump(json_doc, f)
