# imports
import json
import glob
import pymysql as mdb

# constants
file_location = "/home/javaprog/Data/Broad/Richards/richards/richards_dbp.json"
dir_location = "/home/javaprog/Data/Broad/Richards/richards/*.json"
batch_number = 0

# get all the files
file_list = glob.glob(dir_location)

# connect to the database
conn = mdb.connect(host='localhost', user='root', password='this aint no password', charset='utf8', db='richards_gene')
cur = conn.cursor()

sql = """insert into `gene_phenotype` (gene, phenotype, prob)
         values (%s, %s, %s) 
    """

# loop through the files
for file_node in file_list:
    print("============== file: {}".format(file_node))
    
    # load the file
    with open(file_node) as json_file:
        json_data = json.load(json_file)

    # show data
    print("the data is of type {}".format(type(json_data)))
    dataset = json_data.get('phenotype')
    phenotype = json_data.get('phenotype')

    for batch in json_data.get('data'):
        batch_number = batch_number + 1
        gene = batch.get('gene')
        prob = batch.get('prob')
        print("{} - the gene {} has probability {} with phenotype {}".format(batch_number, gene, prob, phenotype))

        cur.execute(sql,(gene, phenotype, prob))
    conn.commit()

    # print("======batch {} of type {}".format(batch_number, type(batch)))
    # for features in batch:
    #     print("features list of size: {} and type {}".format(len(features.get('features')), type(features.get('features'))))


