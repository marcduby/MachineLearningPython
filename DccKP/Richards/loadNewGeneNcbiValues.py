# imports
import pandas as pd 
import pymysql as mdb

# file location
file_loc = "/home/javaprog/Data/Broad/Richards/Gene/gene_info"

# load file into pandas dataframe
gene_df = pd.read_csv (file_loc, sep = '\t')

# filter the dataframe
gene_df = gene_df[gene_df['Symbol'] == gene_df['Symbol_from_nomenclature_authority']]
print("the filtered df has shape of {}".format(gene_df.shape))

# describe the dataframe
print("data: \n{}\n\n".format(gene_df.head(5)))

gene = 'PPARG'
print("gene {} has row \n{}".format(gene, gene_df[gene_df.Symbol == gene]))

# create connection
conn = mdb.connect(host='localhost', user='root', password='this aint no password', charset='utf8', db='genetics_lookup')
cur = conn.cursor()

sql = """insert into `gene_ncbi_load` (ncbi_id_int, ncbi_id, gene)
         values (%s, %s, %s) 
    """

# loop through rows
counter = 0
for index, row in gene_df.iterrows():
    ncbi_id = 'NCBIGene:' + str(row['GeneID'])
    ncbi_id_int = row['GeneID']
    gene_name = row['Symbol']
    
    cur.execute(sql,(ncbi_id_int, ncbi_id, gene_name))
    counter = counter + 1

    # commit every 1000
    if counter % 1000 == 0:
        print("{} - gene {} with id {}".format(counter, ncbi_id, gene_name))
        conn.commit()


