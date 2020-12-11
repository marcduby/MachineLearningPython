# imports
import pandas as pd 
import pymysql as mdb

# file location
file_loc = "/home/javaprog/Data/Broad/ABC/abc.unique.disease.variant_gene.tsv"

# load file into pandas dataframe
gene_df = pd.read_csv (file_loc, sep = '\t')
print("the filtered df has shape of {}".format(gene_df.shape))

# describe the dataframe
print("data: \n{}\n\n".format(gene_df.head(25)))

# create connection
# conn = mdb.connect(host='localhost', user='root', password='this aint no password', charset='utf8', db='tran_genepro')
conn = mdb.connect(host='localhost', user='root', password='yoyoma', charset='utf8', db='tran_genepro')
cur = conn.cursor()

sql = """insert into `abc_gene_phenotype` (phenotype, gene, p_value)
         values (%s, %s, %s) 
    """

# loop through rows
counter = 0
for index, row in gene_df.iterrows():
    cur.execute(sql,(row['phenotype'], row['gene'], row['p-value']))
    counter = counter + 1

    # commit every 10
    if counter % 100 == 0:
        print("{} - gene {} with id {}".format(counter, row['phenotype'], row['gene']))
        conn.commit()

conn.commit()
