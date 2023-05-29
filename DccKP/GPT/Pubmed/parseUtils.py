

# imports
import os


# constants
string_test = """ 
    Sudden Sensorineural Hearing Loss (SSNHL) is a quite common clinical finding in otolaryngology. 
    Most cases are classified as idiopathic and there is a dearth of information on factors able to predict the response to treatment and hearing recovery. 
    The main aim of this systematic review and meta-analysis was to assess and critically discuss the role of circulating inflammatory biomarkers in SSNHL. : 
    A search was conducted of the English literature published between 1 January 2009 and 7 July 2022 on Pubmed, Scopus, Web of Science, ScienceDirect, 
    and Cochrane following PRISMA guidelines. : 
    A total of 256 titles were retrieved from the search. After full-text screening and application of inclusion/exclusion criteria, 13 articles were included. 
    Twelve out of thirteen studies reported significant differences in biomarkers values in SSNHL patients, of which Tumor Necrosis Factor alpha (TNF-\u03b1) 
    and C-reactive Protein (CRP) were the most analyzed. Our meta-analysis for CRP's mean values in SSNHL groups vs. controls showed significantly higher CRP levels with 
    a pooled overall difference of 1.07; confidence interval (CI) at 95%: 0.03; 2.11. For TNF-\u03b1, discordant results were found: three studies showed 
    significantly higher levels in SSNHL patients vs. controls; 
    whereas other three investigations showed lower levels in the SSNHL groups (overall pooled difference 1.97; 95% CI: -0.90; 4.84). 
    A high between-study heterogeneity was found. : This systematic review pointed out that, although there exists a growing literature 
    in the field of circulatory biomarkers identification in SSNHL, there is a high heterogeneity of results and low quality of evidence. 
    CRP resulted to be higher in SSNHL patients than in controls, while TNF-\u03b1 showed more heterogeneous behavior. 
    The data reported herein needs to be confirmed in well-designed prospective multicenter randomized studies, with the objective of 
    improving SSNHL treatment and outcome and thereby reducing the social burden of hearing loss.
    """

# methods
def parseLinesIntoArray(str_input, list_delimiters, list_remove, log=False):
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
            
            # make result new array
            list_result = list_temp

    # return
    return list_result

def create_json_dataset_file(list_input, file_name, log=False):
    pass

def create_conversation_list(list_input, str_start="<start> ", str_end=" <end>", str_bot=" <bot> ", log=False):
    '''
    creates a list of user/bot conversations
    '''
    list_result = []

    # loop through array
    for index, item in enumerate(list_input):
        if index == len(list_input) - 1:
            continue
        str_temp = str_start + list_input[index] + str_bot + list_input[index + 1] + str_end
        list_result.append(str_temp)

    # return
    return list_result


# main
if __name__ == "__main__":
    # test list delimiter
    list_test = parseLinesIntoArray(string_test, ['. ', ';'], [['\n', ""], ['vs.', 'vs']], log=True)
    print("parsing string")
    for item in list_test:
        print(item)

    # create array of conversations
    print("\n\ncreating conversation")
    list_conv = create_conversation_list(list_test, '<start> ', ' <end>', ' <bot>: ')
    for item in list_conv:
        print(item)

