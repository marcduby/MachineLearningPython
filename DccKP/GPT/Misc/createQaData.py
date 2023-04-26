
import os 
import xmltodict
import re
import glob 
import io
import json 

# constants 
FILE_INPUT_JSON = "/home/javaprog/Data/Broad/GPT/Test/dev-v2.0.json"
FILE_OUTPUT_TRAIN_JSON = "/home/javaprog/Data/Broad/GPT/Test/Norman/data_train_norman.json"
FILE_OUTPUT_TEST_JSON = "/home/javaprog/Data/Broad/GPT/Test/Norman/data_test_norman.json"


def parse_qa_json(json_input, log=False):
    '''
    parses the json file into the map we want
    '''
    list_context_sections = []

    # only grab the question, answers and context 
    if json_input.get('data'):
        for paragraph in json_input.get('data'):
            json_root = paragraph.get('paragraphs')
            if json_root:
                for item in json_root:
                    if item.get('context'):
                        list_input_qas = item.get('qas')
                        list_qas = []
                        for qa in list_input_qas:
                            print("\n{}".format(qa))
                            if qa.get('answers') and qa.get('id') and qa.get('question'):
                                # add the question to the list
                                list_qas.append(qa)

                        # add the question list
                        list_context_sections.append({'qas': list_qas, 'context': item.get('context')})

    # return
    return list_context_sections[0: 1000], list_context_sections[1000:]

if __name__ == "__main__":
    json_input = None

    # get the dict of the xml file 
    with open(FILE_INPUT_JSON) as file_input:
        json_input = json.load(file_input)

    # get the 
    json_train_list, json_test_list = parse_qa_json(json_input)

    # print
    print(json_test_list)
    print("got question list of train size: {} and test size: {}".format(len(json_train_list), (len(json_test_list))))

    # save to json
    with open(FILE_OUTPUT_TRAIN_JSON, "w") as f:
        json.dump(json_train_list, f)
    with open(FILE_OUTPUT_TEST_JSON, "w") as f:
        json.dump(json_test_list, f)

