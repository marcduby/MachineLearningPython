

# imports 
import pandas as pd 
import requests 
import time

# constants 
file_rare_disease = '/home/javaprog/Data/Broad/Translator/RareDisease/DCC_GARD_RareDiseases.csv'
file_test_rare_disease = '/home/javaprog/Data/Broad/Translator/RareDisease/Test_DCC_GARD_RareDiseases.csv'
url_name_search = 'https://name-resolution-sri.renci.org/lookup?string={}'

# methods 
def find_ontology(disease):
    '''will call REST api and will return ontology id if name exact match '''
    # initialize
    ontology_id = None

    # call the url
    response = requests.post(url_name_search.format(disease.replace("-", " ")))
    output_json = response.json()

    # loop through results, find first exact result
    for key, values in output_json.items():
        # print("key: {}".format(key))
        # print("value: {}\n".format(values))
        # do MONDO search first since easiest comparison
        if 'MONDO' in key:
            if disease.lower() in map(str.lower, values):
                ontology_id = key
                break

    # return
    return ontology_id

# read the file
df_rare_disease = pd.read_csv(file_rare_disease, sep=',', header=0)
print("after reading: \n{}".format(df_rare_disease.info()))

# loop through rows and look for match for disease name 
count = 0
for index, row in df_rare_disease.iterrows():
    ontology = row['ontology']
    ontology_check = row['ontology_check']
    if pd.isnull(ontology) and pd.isnull(ontology_check):
        # log
        print("no previous ontology for: {}".format(row['d.name']))
        count += 1

        # find ontology
        result = find_ontology(row['d.name'])

        # if found, log and set
        if result is not None:
            print("found ontology for: {} - {}\n".format(row['d.name'], result))
            df_rare_disease.loc[df_rare_disease['d.name'] == row['d.name'], ['ontology']] = result
        
        # log that checked
        df_rare_disease.loc[df_rare_disease['d.name'] == row['d.name'], ['ontology_check']] = "yes"

        # break if count reached
        if count%10 == 0:
            print("{} - data saved to file".format(count))
            df_rare_disease.to_csv(file_rare_disease, sep=',', index=False)
            # break

        # sleep for throttling avoidance
        # time.sleep(10)
    
# log
print("\nafter updating: \n{}".format(df_rare_disease.info()))

# write out results 
# df_rare_disease.to_csv(file_test_rare_disease, sep=',')
df_rare_disease.to_csv(file_rare_disease, sep=',', index=False)
