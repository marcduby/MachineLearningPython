
# imports
import os 
import glob 
import io
import json
import pymysql as mdb

# constants
DB_PASSWD = os.environ.get('DB_PASSWD')
SCHEMA_GPT = "pubmed_gpt"
DIR_DATA = "/home/javaprog/Data/Broad/GPT/TextGenerationBert/Input"
FILE_DATA = "{}/input_bert.txt".format(DIR_DATA)
SQL_SELECT = "select pubmed_id, abstract_text from {}.pmd_abstract limit 20000".format(SCHEMA_GPT)
SQL_WHERE = " where abstract_text like %s"
SQL_WHERE = " where pubmed_id in (36061186,35928446,36072671,36171883,36173399,35910211,36105085,35754818,35480303)"
LIST_REPLACE = [["\n", ""], ["CI.", "CI"], ["#text", ""]]

# methods
def parse_lines_into_array(str_input, list_delimiters, list_remove, num_min_length=40, log=False):
    '''
    will split the input into arrays based on the delimiters provided 
    '''
    list_result = []

    # replace
    for item, rep in list_remove:
        str_input = str_input.replace(item, rep)

    # split on the first character
    if log:
        print("Splitting on '{}'".format(list_delimiters[0]))
    list_result = str_input.split(list_delimiters[0])

    # then if more, delimiters, split on those
    if len(list_delimiters) > 1:
        for delim in list_delimiters[1:]:
            if log:
                print("Splitting on '{}'".format(delim))
            list_temp = []
            for line in list_result:
                list_temp = list_temp + line.split(delim)

            list_result = list_temp

    # skip any senetence that is less then X chars
    list_result = [item for item in list_result if len(item) > num_min_length]

    # return            
    return list_result

def create_text_dataset_file(list_input, file_name, log=False):
    # save to json
    with open(file_name, "w+") as f:
        f.write("\n".join(map(str, list_input)))

    print("wrote out: {} size list to: {}".format(len(list_input), file_name))

def create_conversation_list(list_input, str_start="<start> ", str_end=" <end>", log=False):
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

def get_list_of_abstracts(conn, keyword, log=False):
    '''
    retrieves the list of abtsracts from the database
    '''
    cursor = conn.cursor()
    list_abstract = []

    # run the qery
    if keyword:
        print(SQL_SELECT + SQL_WHERE)
        # cursor.execute(SQL_SELECT + SQL_WHERE, ("'%" + keyword + "%'"))
        cursor.execute(SQL_SELECT + SQL_WHERE)
    else:
        cursor.execute(SQL_SELECT, ())

    # get the results
    db_results = cursor.fetchall()
    for row in db_results:
        print("got pubmed: {}".format(row[0]))
        list_abstract.append(row[1])

    # return
    return list_abstract

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
    list_conversations = []
    num_count = 0

    # get the list of abstracts
    conn = get_connection()
    # list_abstracts = get_list_of_abstracts(conn, 'PCSK9')
    list_abstracts = get_list_of_abstracts(conn, None)
    num_abstracts = len(list_abstracts)
    print("to process, got list of abstracts of size: {}".format(num_abstracts))

    # for each abstract
    for abstract in list_abstracts:
        num_count = num_count + 1
        # split the abstract into sentences
        list_sentences = parse_lines_into_array(abstract, [".", ";"], [["\n", ""], ["vs.", "vs"], ["'#text': ", ""]], log=True)
        # list_sentences = parse_lines_into_array(abstract, [",", ";"], [["\n", ""], ["vs.", "vs"]])

        # append a space for extra line
        list_sentences.append("")

        # log
        print("added: {}/{}".format(num_count, num_abstracts))
        # add the lists to the overal list
        list_conversations = list_conversations + list_sentences

    # log
    for item in list_conversations:
        print(item)

    # write out the conversations
    create_text_dataset_file(list_conversations, FILE_DATA)